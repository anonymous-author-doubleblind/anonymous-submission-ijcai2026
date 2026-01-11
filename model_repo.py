# -*- coding: utf-8 -*-
# pycd/meta/model_repo.py
# 冻结的基模型仓库：统一对外提供 predict_all(x) -> [M1(x),...,MK(x)]

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, glob, json
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# 统一用工厂函数创建，避免各模型构造差异
from pycd.models.init_model import create_model

# 导入图工具函数
from data.graph_utils import construct_local_map, ensure_rcd_graph_files
from data.dataset import load_question_mapping, load_q_matrix, CDMDataset


def _load_ckpt(model: nn.Module, ckpt_path: str, strict: bool = True) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    # 兼容多种保存格式
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=strict)


def _find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _read_model_params_from_dir(ckpt_dir: str) -> Dict[str, Any]:
    """
    优先读取 wandb_train_test.py 保存的 results_fold*.json（其中包含 model_params）；
    若不存在，再尝试 config/params 的 json/yaml。
    """
    # 1) 尝试 results_fold*.json
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "results_fold*.json")))
    for fp in cands[::-1]:  # 最新的优先
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "model_params" in obj and isinstance(obj["model_params"], dict):
                return obj["model_params"]
        except Exception:
            pass

    # 2) 常见命名
    common = [
        "config.json", "params.json", "model_config.json", "experiment_config.json",
        "config.yaml", "config.yml", "params.yaml", "params.yml"
    ]
    for name in common:
        p = os.path.join(ckpt_dir, name)
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                # 简单 YAML 读取（无外部依赖）
                import yaml  # 若环境无 pyyaml，请用 json 代替或删除 yaml 配置
                with open(p, "r", encoding="utf-8") as f:
                    obj = yaml.safe_load(f)
            if isinstance(obj, dict):
                # 可能直接就是 model_params，也可能在某个 key 里
                if "model_params" in obj and isinstance(obj["model_params"], dict):
                    return obj["model_params"]
                return obj
        except Exception:
            continue

    return {}  # 没找到则返回空字典，由上层回退到默认值


def _infer_model_name_from_path(ckpt_path: str) -> Optional[str]:
    base = os.path.basename(os.path.dirname(ckpt_path)).lower()
    for name in ["kancd", "kscd", "rcd", "orcdf", "neuralcdm", "icdm"]:
        if name in base:
            return name
    return None


class ModelRepo(nn.Module):
    """
    管理一组已训练的“冻结”基模型：统一签名
      forward / predict_all(stu_id, exer_id, q_vector) -> [B, K]
    """
    def __init__(
        self,
        specs: List[Dict[str, Any]],
        counts: Tuple[int, int, int],   # (n_students, n_exercises, n_concepts)
        device: torch.device = torch.device("cpu")
    ):
        """
        specs 形如：
          [{'name': 'kancd', 'ckpt': '/.../best_model.pth'}, ...]
        或可带超参覆盖：
          [{'name':'kancd','ckpt':'...','emb_dim':20,'hidden_dims':[256,64],...}]
        name 可省略，将从 ckpt 上级目录名推断（包含 kancd/kscd/rcd/orcdf/neuralcdm 即可）。
        """
        super().__init__()
        n_students, n_exercises, n_concepts = counts
        self.device = device
        self.models = nn.ModuleList()
        self.base_names: List[str] = []

        for sp in specs:
            ckpt = sp.get("ckpt")
            assert ckpt and os.path.exists(ckpt), f"ckpt not found: {ckpt}"
            name = (sp.get("name") or _infer_model_name_from_path(ckpt))
            assert name, f"无法从路径推断模型名，请在 spec 里显式提供 name。ckpt={ckpt}"
            name = name.lower()

            # 读取保存时的 model_params，并与显式覆盖项合并
            ckpt_dir = os.path.dirname(ckpt)
            saved_params = _read_model_params_from_dir(ckpt_dir)

            # 兼容不同命名：有些保存成 hidden_dims / dropout 列表，有些拆成 1/2
            def _norm_params(p: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(p or {})
                # 统一模型名字段
                out["model_name"] = name
                
                # 为所有模型添加 device 参数
                if "device" not in out:
                    out["device"] = str(device)  # 使用 ModelRepo 的设备
                
                # 处理 hidden_dims：将列表转换为 hidden_dims1/2（init_model.py 期望的格式）
                if "hidden_dims" in out and isinstance(out["hidden_dims"], (list, tuple)):
                    h_dims = out["hidden_dims"]
                    if len(h_dims) >= 2:
                        out["hidden_dims1"] = int(h_dims[0])
                        out["hidden_dims2"] = int(h_dims[1])
                        # 移除原始的 hidden_dims，避免冲突
                        out.pop("hidden_dims")
                    elif len(h_dims) == 1:
                        out["hidden_dims1"] = int(h_dims[0])
                        out["hidden_dims2"] = int(h_dims[0])  # 复制相同的值
                        out.pop("hidden_dims")
                # 备选：如果已经有 hidden_dims1/2，确保它们是整数
                else:
                    # 保持原有的 hidden_dims1/2 不变
                    pass
                
                # 处理 dropout：将列表转换为 dropout1/2（init_model.py 期望的格式）
                if "dropout" in out and isinstance(out["dropout"], (list, tuple)):
                    d_vals = out["dropout"]
                    if len(d_vals) >= 2:
                        out["dropout1"] = float(d_vals[0])
                        out["dropout2"] = float(d_vals[1])
                        # 移除原始的 dropout，避免冲突
                        out.pop("dropout")
                    elif len(d_vals) == 1:
                        out["dropout1"] = float(d_vals[0])
                        out["dropout2"] = float(d_vals[0])  # 复制相同的值
                        out.pop("dropout")
                # 备选：如果已经有 dropout1/2，确保它们是浮点数
                else:
                    # 保持原有的 dropout1/2 不变
                    pass
                
                # 为特定模型添加必需的参数
                if name == "rcd" or name == "orcdf":
                    # RCD 和 ORCDF 都需要 dataset 和 base_dir 参数
                    if "dataset" not in out:
                        # 从 ckpt 路径中推断数据集名
                        ckpt_parent = os.path.basename(os.path.dirname(ckpt))
                        # 简单的启发式方法
                        supported_datasets = ["frcsub", "assist2009", "assist2012", "assist2017", "math1", "math2"]
                        found_ds = None
                        for ds in supported_datasets:
                            if ds in ckpt_parent.lower():
                                found_ds = ds
                                break
                        out["dataset"] = found_ds or "frcsub" # 默认 frcsub

                    if "base_dir" not in out:
                        # 设置默认的 base_dir
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        out["base_dir"] = os.path.join(project_root, "data")

                if name == "rcd":
                    # RCD 模型需要 gpu 参数，使用与目标设备匹配的 GPU ID
                    if device.type == 'cuda':
                        # 从设备字符串 "cuda:7" 中提取 GPU ID
                        gpu_id = device.index if device.index is not None else 0
                        out["gpu"] = gpu_id
                    else:
                        out["gpu"] = -1  # CPU 模式
                
                elif name == "orcdf":
                    # ORCDF 模型需要的参数，基于用户实际训练命令
                    out.setdefault("keep_prob", 1.0)
                    out.setdefault("dtype", torch.float64) # 使用实际的 torch.dtype 对象
                    out.setdefault("if_type", "kancd")
                    out.setdefault("weight_decay", 0.0)
                    out.setdefault("mode", "all")
                    # ORCDF 模型需要 graph_dict 参数，创建一个空的字典
                    out.setdefault("graph_dict", {
                        'right': None, 'wrong': None, 'right_flip': None, 'wrong_flip': None,
                        'Q_Matrix': None, 'response': None, 'flip_ratio': out.get("flip_ratio", 0.15)
                    })
                
                elif name == "icdm":
                    # ICDM: 将 gcn_layers 重命名为 gcnlayers（init_model.py 期望的格式）
                    if "gcn_layers" in out:
                        out["gcnlayers"] = out.pop("gcn_layers")
                    # ICDM 需要 graph_dict 参数
                    out.setdefault("graph_dict", {})
                    # ICDM 需要 dataset 和 base_dir 参数（用于构建 norm_adj）
                    if "dataset" not in out:
                        ckpt_parent = os.path.basename(os.path.dirname(ckpt))
                        supported_datasets = ["frcsub", "assist2009", "assist2012", "assist2017", "math1", "math2"]
                        found_ds = None
                        for ds in supported_datasets:
                            if ds in ckpt_parent.lower():
                                found_ds = ds
                                break
                        out["dataset"] = found_ds or "math1"
                    if "base_dir" not in out:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        out["base_dir"] = os.path.join(project_root, "data")
                
                # emb_dim / mf_type 等保持原样
                return out

            params = _norm_params(saved_params)
            # 用户在 specs 里填写的覆盖（优先级最高）
            override = _norm_params({k: v for k, v in sp.items() if k not in {"name", "ckpt"}})
            params.update(override)

            # 构建 args（create_model 内部会根据 model_name 取需要的字段）
            args = Namespace(**params)

            # 用工厂函数构建模型（可跨模型类型，屏蔽构造差异）
            model, model_params, _ = create_model(
                args,
                concept_count=n_concepts,
                exercise_count=n_exercises,
                user_count=n_students
            )

            # 加载权重 & 冻结
            # ICDM 需要特殊处理，因为它不是 nn.Module
            if name != "icdm":
                _load_ckpt(model, ckpt, strict=True)
                model.eval() # 先设置 eval 模式

            # --- 特殊模型初始化适配 ---
            # RCD: 解决设备不匹配问题
            if name == "rcd":
                model.to(device)
                # RCD内部有自己的device逻辑，这里强制覆盖
                model.device = device
                if hasattr(model, 'FusionLayer1'):
                    model.FusionLayer1.device = device
                if hasattr(model, 'FusionLayer2'):
                    model.FusionLayer2.device = device
                
                # 递归地将所有子模块和图移动到正确的设备
                def move_graphs_to_device(module, target_device):
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        attr_value = getattr(module, attr_name)
                        if hasattr(attr_value, 'to') and callable(getattr(attr_value, 'to')) and not isinstance(attr_value, torch.nn.Module):
                            try:
                                setattr(module, attr_name, attr_value.to(target_device))
                            except Exception:
                                pass # 忽略无法移动的属性
                    for child in module.children():
                        move_graphs_to_device(child, target_device)
                
                move_graphs_to_device(model, device)
                print(f"[RCD Adapter] Forced model and its graphs to device: {device}")

            # ORCDF: 复用 wandb_train_test.py 中的图构建逻辑
            elif name == "orcdf":
                model.to(device)
                print("[ORCDF Adapter] Building graph using logic from wandb_train_test.py...")
                try:
                    model.device = str(device)
                    if hasattr(model, 'extractor'):
                        model.extractor.device = str(device)

                    train_valid_csv = os.path.join(params["base_dir"], params["dataset"], "train_valid.csv")
                    if os.path.exists(train_valid_csv):
                        mapping_path = os.path.join(params["base_dir"], params["dataset"], "id_mapping.json")
                        q_matrix_path = os.path.join(params["base_dir"], params["dataset"], "q_matrix.csv")
                        
                        question2idx = load_question_mapping(mapping_path)
                        q_matrix = load_q_matrix(q_matrix_path, None, num_exercises=len(question2idx), num_concepts=None)
                        
                        from pycd.models.orcdf import extract_response_array
                        # 注意：这里不划分 fold，使用全量 train_valid 数据来构建图，与原始训练流程保持一致
                        temp_dataset = CDMDataset(train_valid_csv, question2idx, q_matrix)
                        response_array = extract_response_array(temp_dataset)
                        
                        se_graph_right, se_graph_wrong = [model.extractor._create_adj_se(response_array, is_subgraph=True)[i] for i in range(2)]
                        se_graph = model.extractor._create_adj_se(response_array, is_subgraph=False)
                        
                        graph_dict = {
                            'right': model.extractor._final_graph(se_graph_right, q_matrix),
                            'wrong': model.extractor._final_graph(se_graph_wrong, q_matrix),
                            'response': response_array,
                            'Q_Matrix': q_matrix.copy(),
                            'flip_ratio': model.flip_ratio,
                            'all': model.extractor._final_graph(se_graph, q_matrix)
                        }
                        
                        model.extractor.get_graph_dict(graph_dict)
                        model.extractor.get_flip_graph()
                        print(f"[ORCDF Adapter] Successfully built and set graph structure on device: {device}")
                    else:
                        print(f"[ORCDF Adapter] Warning: {train_valid_csv} not found, graph structure will be incomplete.")
                except Exception as e:
                    print(f"[ORCDF Adapter] CRITICAL: Failed to build graph structure: {e}")

            # ICDM: 特殊初始化逻辑，需要构建 IGNet 并加载权重
            elif name == "icdm":
                print("[ICDM Adapter] Building IGNet and loading weights...")
                try:
                    import scipy.sparse as sp
                    import dgl
                    from pycd.models.icdm import IGNet
                    
                    # 读取训练数据来构建 norm_adj 和图
                    train_valid_csv = os.path.join(params["base_dir"], params["dataset"], "train_valid.csv")
                    q_matrix_path = os.path.join(params["base_dir"], params["dataset"], "q_matrix.csv")
                    mapping_path = os.path.join(params["base_dir"], params["dataset"], "id_mapping.json")
                    
                    if os.path.exists(train_valid_csv):
                        train_df = pd.read_csv(train_valid_csv)
                        train_stu = train_df["user_id"].values
                        # 兼容不同的列名：question_id 或 exercise_id
                        if "question_id" in train_df.columns:
                            train_exer = train_df["question_id"].values
                        else:
                            train_exer = train_df["exercise_id"].values
                        train_correct = train_df["correct"].values
                        
                        # 加载 Q 矩阵
                        question2idx = load_question_mapping(mapping_path)
                        q_matrix = load_q_matrix(q_matrix_path, None, num_exercises=len(question2idx), num_concepts=None)
                        q_tensor = torch.tensor(q_matrix, dtype=torch.float32)
                        
                        # 构建 norm_adj 矩阵
                        def get_adj_matrix(tmp_adj):
                            adj_mat = tmp_adj + tmp_adj.T
                            rowsum = np.array(adj_mat.sum(1))
                            d_inv = np.power(rowsum, -0.5).flatten()
                            d_inv[np.isinf(d_inv)] = 0.
                            d_mat_inv = sp.diags(d_inv)
                            norm_adj_tmp = d_mat_inv.dot(adj_mat)
                            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
                            return adj_matrix

                        def sp_mat_to_sp_tensor(sp_mat):
                            coo = sp_mat.tocoo().astype(np.float64)
                            indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
                            return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

                        n_nodes = n_students + n_exercises
                        ratings = np.ones_like(train_stu, dtype=np.float64)
                        tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + n_students)), shape=(n_nodes, n_nodes))
                        norm_adj = sp_mat_to_sp_tensor(get_adj_matrix(tmp_adj))
                        
                        # 构建 ICDM 需要的图结构
                        # 1. Q 图 (题目-知识点)
                        q_np = q_matrix
                        node_q = n_exercises + n_concepts
                        g_q = dgl.DGLGraph()
                        g_q.add_nodes(node_q)
                        edge_list_q = []
                        indices = np.where(q_np != 0)
                        for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
                            edge_list_q.append((int(know_id + n_exercises), int(exer_id)))
                            edge_list_q.append((int(exer_id), int(know_id + n_exercises)))
                        if edge_list_q:
                            src_q, dst_q = tuple(zip(*edge_list_q))
                            g_q.add_edges(src_q, dst_q)
                        
                        # 2. right/wrong 图 (学生-题目正确/错误交互)
                        node_se = n_students + n_exercises
                        g_right = dgl.DGLGraph()
                        g_right.add_nodes(node_se)
                        g_wrong = dgl.DGLGraph()
                        g_wrong.add_nodes(node_se)
                        right_edge_list = []
                        wrong_edge_list = []
                        for idx in range(len(train_stu)):
                            stu_id = int(train_stu[idx])
                            exer_id = int(train_exer[idx])
                            correct = int(float(train_correct[idx]))
                            if correct == 1:
                                right_edge_list.append((stu_id, exer_id + n_students))
                                right_edge_list.append((exer_id + n_students, stu_id))
                            else:
                                wrong_edge_list.append((stu_id, exer_id + n_students))
                                wrong_edge_list.append((exer_id + n_students, stu_id))
                        if right_edge_list:
                            right_src, right_dst = tuple(zip(*right_edge_list))
                            g_right.add_edges(right_src, right_dst)
                        if wrong_edge_list:
                            wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
                            g_wrong.add_edges(wrong_src, wrong_dst)
                        
                        # 3. I 图 (学生-知识点涉及)
                        node_sc = n_students + n_concepts
                        g_i = dgl.DGLGraph()
                        g_i.add_nodes(node_sc)
                        edge_list_i = []
                        sc_matrix = np.zeros(shape=(n_students, n_concepts))
                        for idx in range(len(train_stu)):
                            stu_id = int(train_stu[idx])
                            exer_id = int(train_exer[idx])
                            concepts = np.where(q_np[exer_id] != 0)[0]
                            for concept_id in concepts:
                                if sc_matrix[stu_id, concept_id] != 1:
                                    edge_list_i.append((stu_id, concept_id + n_students))
                                    edge_list_i.append((concept_id + n_students, stu_id))
                                    sc_matrix[stu_id, concept_id] = 1
                        if edge_list_i:
                            src_i, dst_i = tuple(zip(*edge_list_i))
                            g_i.add_edges(src_i, dst_i)
                        
                        # 组装图字典
                        icdm_graph = {
                            'Q': g_q,
                            'right': g_right,
                            'wrong': g_wrong,
                            'I': g_i
                        }
                        
                        # 创建 IGNet 网络
                        icdm_net = IGNet(
                            stu_num=n_students,
                            prob_num=n_exercises,
                            know_num=n_concepts,
                            dim=params.get("dim", 32),
                            graph=icdm_graph,
                            norm_adj=norm_adj,
                            device=str(device),
                            gcnlayers=params.get("gcnlayers", 3),
                            agg_type=params.get("agg_type", "mean"),
                            cdm_type=params.get("cdm_type", "glif"),
                            khop=params.get("khop", 2)
                        ).to(device)
                        
                        # 加载权重到 IGNet
                        sd = torch.load(ckpt, map_location="cpu")
                        if isinstance(sd, dict) and "state_dict" in sd:
                            sd = sd["state_dict"]
                        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                            sd = sd["model"]
                        icdm_net.load_state_dict(sd, strict=True)
                        icdm_net = icdm_net.to(torch.float64)
                        icdm_net.eval()
                        
                        # 将 IGNet 赋值给 ICDM.net
                        model.net = icdm_net
                        print(f"[ICDM Adapter] Successfully initialized IGNet on device: {device}")
                    else:
                        raise FileNotFoundError(f"train_valid.csv not found at {train_valid_csv}")
                except Exception as e:
                    print(f"[ICDM Adapter] CRITICAL: Failed to initialize IGNet: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            # 通用收尾工作
            if name != "icdm":
                model.to(device) # 再次确保整个模型在目标设备上
                for p in model.parameters():
                    p.requires_grad_(False)
                self.models.append(model)
            else:
                # ICDM 的实际网络是 model.net (IGNet)，直接添加它
                for p in model.net.parameters():
                    p.requires_grad_(False)
                self.models.append(model.net)  # 添加 IGNet 而不是 ICDM

            self.base_names.append(name)

        self.n_bases = len(self.models)

    @torch.no_grad()
    def predict_all(self, stu_id: torch.LongTensor, exer_id: torch.LongTensor, q_vector: torch.Tensor) -> torch.Tensor:
        """
        返回 [B, K]：第 k 列为第 k 个基模型的概率输出。
        """
        preds = []
        for i, m in enumerate(self.models):
            model_name = self.base_names[i]
            
            # --- Adapter Logic ---
            if model_name == 'rcd':
                # RCD 使用 kn_r 参数
                y = m(stu_id, exer_id, kn_r=q_vector)
            elif model_name == 'orcdf':
                # ORCDF 使用不同的参数名，并返回一个元组 (float64)
                y_tuple = m(user_id=stu_id, item_id=exer_id, q_matrix=q_vector)
                y = y_tuple[0].float() # 只取 logits，并转换为 float32
                y = torch.sigmoid(y_tuple[0].float())
            elif model_name == 'icdm':
                # ICDM 已经是 IGNet，直接调用（输出是 float64，需转换为 float32）
                y = m(stu_id, exer_id, q_vector)
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()
                else:
                    y = y.float()  # 将 double 转换为 float32
            else:
                # 标准模型
                y = m(stu_id, exer_id, q_vector)
            
            if y.dim() == 1:
                y = y.unsqueeze(1)
            preds.append(y)
        return torch.cat(preds, dim=1)

    # 兼容直接 forward 调用
    def forward(self, stu_id, exer_id, q_vector):
        return self.predict_all(stu_id, exer_id, q_vector)
