# -*- coding: utf-8 -*-
# pycd/meta/model_repo.py
# Frozen base-model repository: expose a unified API predict_all(x) -> [M1(x),...,MK(x)]

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, glob, json
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Use a factory function to create models to avoid constructor differences across model types
from cd.models.init_model import create_model

# Graph utility helpers
from data.graph_utils import construct_local_map, ensure_rcd_graph_files
from data.dataset import load_question_mapping, load_q_matrix, CDMDataset


def _load_ckpt(model: nn.Module, ckpt_path: str, strict: bool = True) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    # Support multiple checkpoint formats
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
    Prefer reading results_fold*.json saved by wandb_train_test.py (contains model_params);
    if not found, fall back to common config/params json/yaml files.
    """
    # 1) Try results_fold*.json
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "results_fold*.json")))
    for fp in cands[::-1]:  # prefer the latest
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "model_params" in obj and isinstance(obj["model_params"], dict):
                return obj["model_params"]
        except Exception:
            pass

    # 2) Common filenames
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
                # Simple YAML read (requires PyYAML)
                import yaml  # If PyYAML is unavailable, prefer JSON configs or remove YAML usage.
                with open(p, "r", encoding="utf-8") as f:
                    obj = yaml.safe_load(f)
            if isinstance(obj, dict):
                # It may be the model_params dict itself, or nested under a key.
                if "model_params" in obj and isinstance(obj["model_params"], dict):
                    return obj["model_params"]
                return obj
        except Exception:
            continue

    return {}  # Not found -> return empty dict; caller should fall back to defaults.


def _infer_model_name_from_path(ckpt_path: str) -> Optional[str]:
    base = os.path.basename(os.path.dirname(ckpt_path)).lower()
    for name in ["kancd", "kscd", "rcd", "orcdf", "neuralcdm", "icdm"]:
        if name in base:
            return name
    return None


class ModelRepo(nn.Module):
    """
    Manage a set of trained "frozen" base models with a unified signature:
      forward / predict_all(stu_id, exer_id, q_vector) -> [B, K]
    """
    def __init__(
        self,
        specs: List[Dict[str, Any]],
        counts: Tuple[int, int, int],   # (n_students, n_exercises, n_concepts)
        device: torch.device = torch.device("cpu")
    ):
        """
        specs examples:
          [{'name': 'kancd', 'ckpt': '/.../best_model.pth'}, ...]
        You may also include hyperparameter overrides:
          [{'name':'kancd','ckpt':'...','emb_dim':20,'hidden_dims':[256,64],...}]
        name can be omitted and will be inferred from the checkpoint's parent directory name
        (must contain one of: kancd/kscd/rcd/orcdf/neuralcdm/icdm).
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
            assert name, f"Cannot infer model name from path; please provide 'name' in spec explicitly. ckpt={ckpt}"
            name = name.lower()

            # Read saved model_params and merge with explicit overrides
            ckpt_dir = os.path.dirname(ckpt)
            saved_params = _read_model_params_from_dir(ckpt_dir)

            # Normalize parameters across different naming conventions:
            # some runs save hidden_dims/dropout as lists; others split them into 1/2.
            def _norm_params(p: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(p or {})
                # Normalize model name field
                out["model_name"] = name
                
                # Add device for all models if missing
                if "device" not in out:
                    out["device"] = str(device)  # use ModelRepo device
                
                # Handle hidden_dims: convert list -> hidden_dims1/2 (init_model.py expects this format)
                if "hidden_dims" in out and isinstance(out["hidden_dims"], (list, tuple)):
                    h_dims = out["hidden_dims"]
                    if len(h_dims) >= 2:
                        out["hidden_dims1"] = int(h_dims[0])
                        out["hidden_dims2"] = int(h_dims[1])
                        # Remove original hidden_dims to avoid conflicts
                        out.pop("hidden_dims")
                    elif len(h_dims) == 1:
                        out["hidden_dims1"] = int(h_dims[0])
                        out["hidden_dims2"] = int(h_dims[0])  # duplicate the same value
                        out.pop("hidden_dims")
                # Else: if hidden_dims1/2 already exist, keep them as-is
                else:
                    pass
                
                # Handle dropout: convert list -> dropout1/2 (init_model.py expects this format)
                if "dropout" in out and isinstance(out["dropout"], (list, tuple)):
                    d_vals = out["dropout"]
                    if len(d_vals) >= 2:
                        out["dropout1"] = float(d_vals[0])
                        out["dropout2"] = float(d_vals[1])
                        # Remove original dropout to avoid conflicts
                        out.pop("dropout")
                    elif len(d_vals) == 1:
                        out["dropout1"] = float(d_vals[0])
                        out["dropout2"] = float(d_vals[0])  # duplicate the same value
                        out.pop("dropout")
                # Else: if dropout1/2 already exist, keep them as-is
                else:
                    pass
                
                # Add required params for specific models
                if name == "rcd" or name == "orcdf":
                    # RCD and ORCDF both need dataset and base_dir
                    if "dataset" not in out:
                        # Infer dataset name from checkpoint path
                        ckpt_parent = os.path.basename(os.path.dirname(ckpt))
                        # Simple heuristic
                        supported_datasets = ["frcsub", "assist2009", "assist2012", "assist2017", "math1", "math2"]
                        found_ds = None
                        for ds in supported_datasets:
                            if ds in ckpt_parent.lower():
                                found_ds = ds
                                break
                        out["dataset"] = found_ds or "frcsub"  # default: frcsub

                    if "base_dir" not in out:
                        # Default base_dir
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        out["base_dir"] = os.path.join(project_root, "data")

                if name == "rcd":
                    # RCD needs a gpu param; use a GPU ID consistent with the target device
                    if device.type == 'cuda':
                        # Extract GPU ID from device string like "cuda:7"
                        gpu_id = device.index if device.index is not None else 0
                        out["gpu"] = gpu_id
                    else:
                        out["gpu"] = -1  # CPU mode
                
                elif name == "orcdf":
                    # ORCDF required params (based on the actual training command)
                    out.setdefault("keep_prob", 1.0)
                    out.setdefault("dtype", torch.float64)  # use an actual torch.dtype object
                    out.setdefault("if_type", "kancd")
                    out.setdefault("weight_decay", 0.0)
                    out.setdefault("mode", "all")
                    # ORCDF expects graph_dict; provide an empty/default one
                    out.setdefault("graph_dict", {
                        'right': None, 'wrong': None, 'right_flip': None, 'wrong_flip': None,
                        'Q_Matrix': None, 'response': None, 'flip_ratio': out.get("flip_ratio", 0.15)
                    })
                
                elif name == "icdm":
                    # ICDM: rename gcn_layers -> gcnlayers (init_model.py expects this format)
                    if "gcn_layers" in out:
                        out["gcnlayers"] = out.pop("gcn_layers")
                    # ICDM expects graph_dict
                    out.setdefault("graph_dict", {})
                    # ICDM needs dataset and base_dir (to build norm_adj)
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
                
                # Keep emb_dim / mf_type, etc. as-is
                return out

            params = _norm_params(saved_params)
            # Overrides specified in specs (highest priority)
            override = _norm_params({k: v for k, v in sp.items() if k not in {"name", "ckpt"}})
            params.update(override)

            # Build args (create_model will pick required fields by model_name)
            args = Namespace(**params)

            # Build model via factory (works across model types; hides constructor differences)
            model, model_params, _ = create_model(
                args,
                concept_count=n_concepts,
                exercise_count=n_exercises,
                user_count=n_students
            )

            # Load weights & freeze
            # ICDM needs special handling because it is not an nn.Module
            if name != "icdm":
                _load_ckpt(model, ckpt, strict=True)
                model.eval()  # eval mode first

            # --- Special model initialization adapters ---
            # RCD: handle device mismatch issues
            if name == "rcd":
                model.to(device)
                # RCD has its own device logic; force override here
                model.device = device
                if hasattr(model, 'FusionLayer1'):
                    model.FusionLayer1.device = device
                if hasattr(model, 'FusionLayer2'):
                    model.FusionLayer2.device = device
                
                # Recursively move all submodules and graphs to the target device
                def move_graphs_to_device(module, target_device):
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        attr_value = getattr(module, attr_name)
                        if hasattr(attr_value, 'to') and callable(getattr(attr_value, 'to')) and not isinstance(attr_value, torch.nn.Module):
                            try:
                                setattr(module, attr_name, attr_value.to(target_device))
                            except Exception:
                                pass  # ignore attributes that cannot be moved
                    for child in module.children():
                        move_graphs_to_device(child, target_device)
                
                move_graphs_to_device(model, device)
                print(f"[RCD Adapter] Forced model and its graphs to device: {device}")

            # ORCDF: reuse graph-building logic from wandb_train_test.py
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
                        # Note: do not split by fold here; use the full train_valid to build graphs,
                        # consistent with the original training pipeline.
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

            # ICDM: special initialization logic; need to build IGNet and load weights
            elif name == "icdm":
                print("[ICDM Adapter] Building IGNet and loading weights...")
                try:
                    import scipy.sparse as sp
                    import dgl
                    from pycd.models.icdm import IGNet
                    
                    # Read training data to build norm_adj and graphs
                    train_valid_csv = os.path.join(params["base_dir"], params["dataset"], "train_valid.csv")
                    q_matrix_path = os.path.join(params["base_dir"], params["dataset"], "q_matrix.csv")
                    mapping_path = os.path.join(params["base_dir"], params["dataset"], "id_mapping.json")
                    
                    if os.path.exists(train_valid_csv):
                        train_df = pd.read_csv(train_valid_csv)
                        train_stu = train_df["user_id"].values
                        # Support different column names: question_id or exercise_id
                        if "question_id" in train_df.columns:
                            train_exer = train_df["question_id"].values
                        else:
                            train_exer = train_df["exercise_id"].values
                        train_correct = train_df["correct"].values
                        
                        # Load Q-matrix
                        question2idx = load_question_mapping(mapping_path)
                        q_matrix = load_q_matrix(q_matrix_path, None, num_exercises=len(question2idx), num_concepts=None)
                        q_tensor = torch.tensor(q_matrix, dtype=torch.float32)
                        
                        # Build norm_adj matrix
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
                        
                        # Build graph structures required by ICDM
                        # 1) Q graph (exercise <-> concept)
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
                        
                        # 2) right/wrong graphs (student <-> exercise interactions by correctness)
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
                        
                        # 3) I graph (student <-> concept involvement)
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
                        
                        # Assemble graph dict
                        icdm_graph = {
                            'Q': g_q,
                            'right': g_right,
                            'wrong': g_wrong,
                            'I': g_i
                        }
                        
                        # Create IGNet network
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
                        
                        # Load weights into IGNet
                        sd = torch.load(ckpt, map_location="cpu")
                        if isinstance(sd, dict) and "state_dict" in sd:
                            sd = sd["state_dict"]
                        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                            sd = sd["model"]
                        icdm_net.load_state_dict(sd, strict=True)
                        icdm_net = icdm_net.to(torch.float64)
                        icdm_net.eval()
                        
                        # Assign IGNet to ICDM.net
                        model.net = icdm_net
                        print(f"[ICDM Adapter] Successfully initialized IGNet on device: {device}")
                    else:
                        raise FileNotFoundError(f"train_valid.csv not found at {train_valid_csv}")
                except Exception as e:
                    print(f"[ICDM Adapter] CRITICAL: Failed to initialize IGNet: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            # Common finalization
            if name != "icdm":
                model.to(device)  # ensure the entire model is on the target device
                for p in model.parameters():
                    p.requires_grad_(False)
                self.models.append(model)
            else:
                # ICDM's actual network is model.net (IGNet); add it directly
                for p in model.net.parameters():
                    p.requires_grad_(False)
                self.models.append(model.net)  # add IGNet rather than ICDM

            self.base_names.append(name)

        self.n_bases = len(self.models)

    @torch.no_grad()
    def predict_all(self, stu_id: torch.LongTensor, exer_id: torch.LongTensor, q_vector: torch.Tensor) -> torch.Tensor:
        """
        Return [B, K], where column k is the probability output of the k-th base model.
        """
        preds = []
        for i, m in enumerate(self.models):
            model_name = self.base_names[i]
            
            # --- Adapter Logic ---
            if model_name == 'rcd':
                # RCD uses kn_r parameter name
                y = m(stu_id, exer_id, kn_r=q_vector)
            elif model_name == 'orcdf':
                # ORCDF uses different param names and returns a tuple (float64)
                y_tuple = m(user_id=stu_id, item_id=exer_id, q_matrix=q_vector)
                y = y_tuple[0].float()  # take logits and convert to float32
                y = torch.sigmoid(y_tuple[0].float())
            elif model_name == 'icdm':
                # ICDM is already IGNet; call directly (output is float64 -> convert to float32)
                y = m(stu_id, exer_id, q_vector)
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()
                else:
                    y = y.float()  # convert double -> float32
            else:
                # Standard models
                y = m(stu_id, exer_id, q_vector)
            
            if y.dim() == 1:
                y = y.unsqueeze(1)
            preds.append(y)
        return torch.cat(preds, dim=1)

    # Compatibility with direct forward calls
    def forward(self, stu_id, exer_id, q_vector):
        return self.predict_all(stu_id, exer_id, q_vector)
