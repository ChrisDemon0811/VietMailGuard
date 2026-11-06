"""Isolated SVD/PCA/LDA research on fixed Development Train/Validation only."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from vietmailguard.embedding_modeling import EmbeddingClassifierArtifact
from .development_training import sha256_file, verify_and_load_development
from .multilingual_dl_experiment import (
    CLASSES, ExperimentBlocked, _json, _smoke_indices, embedding_matrix, evaluate, offline_boundary,
)
from .paths import resolve_v21_paths

EXPERIMENT = "dimensionality_reduction_02"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def directories(paths):
    return paths.model / "experiments" / EXPERIMENT, paths.results / "experiments" / EXPERIMENT


def load_config(paths):
    config = read_json(paths.config / "experiments" / f"{EXPERIMENT}.json")
    if (config["experiment"] != EXPERIMENT or config["scope"] != "development_validation_only"
            or config["download_allowed"] is not False or config["final_test_allowed"] is not False
            or config["runtime_modified"] is not False or config["lda"]["n_components"] > len(CLASSES)-1):
        raise ExperimentBlocked("Configuration violates the isolated development-only contract")
    return config


def memory_snapshot():
    """Native OS measurements; no psutil dependency or torch/encoder allocation."""
    result = {"free_ram_gib": None, "rss_gib": None, "process_peak_rss_gib": None}
    if sys.platform == "win32":
        import ctypes as c
        from ctypes import wintypes as w
        class Memory(c.Structure):
            _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [(name,c.c_ulonglong) for name in (
                "total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        state = Memory()
        state.length = c.sizeof(state)
        if c.windll.kernel32.GlobalMemoryStatusEx(c.byref(state)):
            result.update(free_ram_gib=state.available/2**30, total_ram_gib=state.total/2**30)
        class ProcessMemory(c.Structure):
            _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [(name,c.c_size_t) for name in (
                "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "page", "peak_page")]
        counter = ProcessMemory()
        counter.cb = c.sizeof(counter)
        get_process = c.windll.kernel32.GetCurrentProcess
        get_process.restype = w.HANDLE
        get_memory = c.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [w.HANDLE,c.c_void_p,w.DWORD]
        if get_memory(get_process(),c.byref(counter),counter.cb):
            result.update(rss_gib=counter.working/2**30, process_peak_rss_gib=counter.peak/2**30)
    elif Path("/proc/meminfo").is_file():
        info = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["free_ram_gib"] = int(info["MemAvailable"].split()[0])*1024/2**30
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_gib"] = int(line.split()[1])*1024/2**30
    return result


class MemoryMonitor:
    """100-ms sampled peak RSS for this stage; native process peak reported separately."""
    def __enter__(self):
        self.before = memory_snapshot()
        self.samples = [self.before["rss_gib"]]
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.1):
                self.samples.append(memory_snapshot()["rss_gib"])
        self.thread = threading.Thread(target=sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()
        self.after = memory_snapshot()
        values = [v for v in self.samples+[self.after["rss_gib"]] if v is not None]
        self.result = {"before":self.before,"after":self.after,
            "sampled_peak_rss_gib":max(values) if values else None,"sampling_interval_ms":100}


def require_ram(additional_bytes, config):
    snapshot = memory_snapshot()
    required = additional_bytes/2**30*config["workspace_safety_factor"] + config["ram_headroom_gib"]
    if snapshot["free_ram_gib"] is None or snapshot["free_ram_gib"] < required:
        raise ExperimentBlocked(f"RAM gate: need {required:.3f} GiB available, measured {snapshot['free_ram_gib']}")
    return {"required_free_gib":required,"measured":snapshot}


def workspace_bytes(family, rows, features, components, validation_rows=0):
    # Additional workspace above resident inputs, float32/float64 conservatively.
    if family == "svd":
        rank = components+10
        return 4*(features*(2*rank+components)+rows*(2*rank+components)+validation_rows*components)
    return 8*(rows*features*3 + features*features*4 + (rows+validation_rows)*components)


def fingerprint(paths):
    manifest_path = paths.splits / "development_bootstrap/manifest.json"
    manifest = read_json(manifest_path)
    runtime_path = paths.config / "runtime_model.json"
    runtime = read_json(runtime_path)
    for path_key, hash_key in (("artifact","artifact_sha256"),("validation_metrics","validation_metrics_sha256"),
                                ("training_config","training_config_sha256")):
        if sha256_file(paths.project_root/runtime[path_key]) != runtime[hash_key]:
            raise ExperimentBlocked("Frozen V2.1 baseline dependency checksum mismatch")
    result = {"config_sha256":sha256_file(paths.config/"experiments"/f"{EXPERIMENT}.json"),
        "development_manifest_sha256":sha256_file(manifest_path),
        "train_sha256":manifest["output_sha256"]["train"],"validation_sha256":manifest["output_sha256"]["validation"],
        "baseline_sha256":runtime["artifact_sha256"],"runtime_config_sha256":sha256_file(runtime_path)}
    return result,runtime


def load_development(paths,config):
    manifest,train,validation = verify_and_load_development(paths,config["max_input_characters"])
    signature,runtime = fingerprint(paths)
    return manifest,train,validation,signature,runtime


def candidate_names(config):
    return ([f"B_svd_{k}" for k in config["svd_components"]]
        + ["C_embedding_384_control"] + [f"C_pca_{k}" for k in config["pca_components"]]
        + ["D_lda_classifier","D_lda2_svc"])


def sparse_bytes(matrix):
    return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes


def sparse_transform(features,frame,batch):
    blocks = []
    for start in range(0,len(frame),batch):
        block = features.transform(frame.iloc[start:start+batch][["subject","body"]])
        if not sparse.issparse(block):
            raise ExperimentBlocked("TF-IDF ceased to be sparse before TruncatedSVD")
        blocks.append(block.tocsr())
    return sparse.vstack(blocks,format="csr",dtype=np.float32)


def prepare_tfidf(paths,config,train,validation,baseline,signature):
    directory = paths.cache / "experiments" / EXPERIMENT
    manifest_path = directory / "tfidf_manifest.json"
    if manifest_path.exists():
        receipt = read_json(manifest_path)
        if receipt["fingerprint"] != signature:
            raise ExperimentBlocked("Sparse feature cache belongs to different inputs")
        if all(sha256_file(directory/f"{split}.npz") == receipt[split]["sha256"] for split in ("train","validation")):
            return receipt
        raise ExperimentBlocked("Sparse feature cache checksum mismatch")
    features = baseline[:-1]
    sample_rows = min(config["transform_batch_rows"],len(train))
    sample = features.transform(train.iloc[:sample_rows][["subject","body"]])
    if not sparse.issparse(sample):
        raise ExperimentBlocked("Frozen TF-IDF features must be sparse")
    estimated = sparse_bytes(sample)*(len(train)+len(validation))/sample_rows
    gate = require_ram(estimated*2.2,config)
    directory.mkdir(parents=True,exist_ok=True)
    receipt = {"fingerprint":signature,"resource_gate":gate,"status":"PASS"}
    for split,frame in (("train",train),("validation",validation)):
        started = time.perf_counter()
        matrix = sparse_transform(features,frame,config["transform_batch_rows"])
        path = directory/f"{split}.npz"
        sparse.save_npz(path,matrix,compressed=False)
        receipt[split] = {"shape":list(matrix.shape),"dtype":str(matrix.dtype),"nnz":int(matrix.nnz),
            "sparse_bytes":sparse_bytes(matrix),"dense_float32_bytes_avoided":int(np.prod(matrix.shape))*4,
            "sha256":sha256_file(path),"transform_and_save_seconds":time.perf_counter()-started}
        del matrix
        gc.collect()
    _json(manifest_path,receipt)
    return receipt


def latency(predict,sample,repeats,scope):
    predict(sample)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(sample)
        timings.append((time.perf_counter()-start)*1000/len(sample))
    return {"median_ms_per_email":float(np.median(timings)),"batch_ms_per_email":timings,
        "rows":len(sample),"repeats":repeats,"scope":scope}


def paired_errors(truth,reference,prediction):
    return {"baseline_correct_candidate_wrong":int(np.sum((reference==truth)&(prediction!=truth))),
        "baseline_wrong_candidate_correct":int(np.sum((reference!=truth)&(prediction==truth)))}


def class_energy(original,reduced,labels,mean=None):
    """Aggregate squared-energy loss; never densifies sparse TF-IDF."""
    if sparse.issparse(original):
        total = np.asarray(original.multiply(original).sum(axis=1)).ravel()
    else:
        total = np.sum(np.square(original if mean is None else original-mean),axis=1)
    retained = np.sum(np.square(reduced),axis=1)
    ratio = np.divide(retained,total,out=np.zeros_like(retained),where=total>0)
    return {label:float(np.mean(1-ratio[labels==label])) if np.any(labels==label) else None for label in CLASSES}


def prepare(paths=None):
    paths = paths or resolve_v21_paths()
    config = load_config(paths)
    _,result_dir = directories(paths)
    with offline_boundary(paths.project_root),threadpool_limits(config["cpu_threads"]):
        manifest,train,validation,signature,runtime = load_development(paths,config)
        result_dir.mkdir(parents=True,exist_ok=True)
        manifest_path = result_dir/"run_manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != signature:
            raise ExperimentBlocked("Experiment already exists with different data/config")
        _json(manifest_path,{"fingerprint":signature,"seed":config["seed"],"scope":config["scope"],
            "leakage":manifest["leakage"],"train_rows":len(train),"validation_rows":len(validation),
            "distribution":{split:frame.groupby(["language","data_origin","label"]).size().reset_index(name="rows").to_dict("records")
                for split,frame in (("train",train),("validation",validation))},
            "environment":{"python":sys.version,"os":platform.platform(),"cpu_count":os.cpu_count(),
                "packages":{name:importlib.metadata.version(name) for name in (
                    "numpy","scipy","scikit-learn","joblib","matplotlib","threadpoolctl")}},
            "config":config,"final_test_opened":False,"runtime_modified":False})
        reference_path = result_dir/"A_baseline.json"
        with MemoryMonitor() as monitor:
            baseline = joblib.load(paths.project_root/runtime["artifact"])
            start = time.perf_counter()
            prediction = np.asarray(baseline.predict(validation[["subject","body"]]))
            seconds = time.perf_counter()-start
            metrics = evaluate(validation,prediction)
            saved = read_json(paths.project_root/runtime["validation_metrics"])
            prior = next(item["metrics"]["macro_f1"] for item in saved["candidates"] if item["candidate"]==runtime["candidate_id"])
            if not np.isclose(metrics["macro_f1"],prior,atol=1e-12,rtol=0):
                raise ExperimentBlocked("Baseline differs from saved Development Validation")
            indices = _smoke_indices(validation,config)
            timing = latency(baseline.predict,validation.iloc[indices][["subject","body"]],config["benchmark_repeats"],"text_pipeline")
        record = {"candidate":"A_baseline","status":"PASS","fingerprint":signature,"metrics":metrics,
            "artifact":runtime["artifact"],"artifact_sha256":runtime["artifact_sha256"],
            "artifact_bytes":(paths.project_root/runtime["artifact"]).stat().st_size,
            "fit_seconds":None,"validation_predict_seconds":seconds,"latency":timing,"memory":monitor.result,
            "reused_frozen_artifact":True,"dimensions":int(baseline.named_steps["classifier"].coef_.shape[1])}
        np.save(result_dir/"baseline_validation_predictions.npy",prediction.astype("U8"),allow_pickle=False)
        record["validation_predictions_sha256"] = sha256_file(result_dir/"baseline_validation_predictions.npy")
        _json(reference_path,record)
        try:
            print(json.dumps({"stage":"sparse_features","rows":len(train)+len(validation)}),flush=True)
            receipt = prepare_tfidf(paths,config,train,validation,baseline,signature)
        except (ExperimentBlocked,MemoryError) as error:
            receipt = {"status":"BLOCKED","reason":str(error) if isinstance(error,ExperimentBlocked) else "MemoryError"}
        _json(result_dir/"tfidf_preparation.json",receipt)
        return {"baseline_macro_f1":metrics["macro_f1"],"tfidf":receipt}


def verified_embeddings(paths,config,train,validation,signature):
    receipt_path = paths.results/"experiments/multilingual_dl_01/embedding_verification.json"
    receipt = read_json(receipt_path)
    if any(receipt["fingerprint"][f"{split}_sha256"] != signature[f"{split}_sha256"] for split in ("train","validation")):
        raise ExperimentBlocked("Previously encoder-verified cache uses different development inputs")
    matrices,info = {},{}
    for split,frame in (("train",train),("validation",validation)):
        matrices[split],info[split] = embedding_matrix(frame,split,paths,config["encoder"],None,"cpu")
        old = receipt["cache"][split]
        if (not old["fresh_encoder_sample_verified"] or old["sha256"]!=info[split]["sha256"]
                or old["manifest_sha256"]!=info[split]["manifest_sha256"]):
            raise ExperimentBlocked("Existing embeddings changed since encoder verification")
    return matrices,{"encoder_verification_sha256":sha256_file(receipt_path),"splits":info,"encoder":config["encoder"]}


def make_reducer(family,components,config):
    if family=="svd":
        return TruncatedSVD(n_components=components,random_state=config["seed"],**config["svd"])
    if family=="pca":
        return PCA(n_components=components,random_state=config["seed"],**config["pca"])
    if family=="lda":
        if components>len(CLASSES)-1:
            raise ExperimentBlocked("LDA dimension exceeds classes minus one")
        return LinearDiscriminantAnalysis(**{**config["lda"],"n_components":components})
    raise ValueError("Unknown reduction family")


def fit_projection(reducer,train,validation,train_labels):
    start = time.perf_counter()
    projected_train = reducer.fit_transform(train,train_labels) if isinstance(reducer,LinearDiscriminantAnalysis) else reducer.fit_transform(train)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    projected_validation = reducer.transform(validation)
    return projected_train,projected_validation,fit_seconds,time.perf_counter()-start


def lda_figure(paths,config,projected_train,projected_validation,train_labels,validation_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    generator = np.random.RandomState(config["seed"])
    figure,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained",sharex=True,sharey=True)
    for axis,data,labels,title in zip(axes,(projected_train,projected_validation),(train_labels,validation_labels),("Train (axes fitted here)","Development Validation"),strict=True):
        for label,color in zip(CLASSES,("#2563eb","#d97706","#dc2626"),strict=True):
            indices = np.flatnonzero(labels==label)
            chosen = generator.choice(indices,min(len(indices),config["plot_max_rows_per_class"]),replace=False)
            axis.scatter(data[chosen,0],data[chosen,1],s=7,alpha=.35,color=color,label=f"{label} (shown {len(chosen)}/{len(indices)})",rasterized=True)
        axis.set(xlabel="Linear discriminant 1",ylabel="Linear discriminant 2",title=title)
        axis.legend(fontsize=8)
    _,directory = directories(paths)
    figure.savefig(directory/"lda_projection.png",dpi=180)
    figure.savefig(directory/"lda_projection.svg")
    plt.close(figure)


def save_artifact(path,artifact):
    if path.exists():
        raise ExperimentBlocked("Refusing to overwrite an experiment artifact")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".partial.joblib")
    joblib.dump(artifact,temporary)
    temporary.replace(path)
    return joblib.load(path,mmap_mode="r")


def candidate(name,paths=None):
    paths = paths or resolve_v21_paths()
    config = load_config(paths)
    model_dir,result_dir = directories(paths)
    result_path = result_dir/f"{name}.json"
    with offline_boundary(paths.project_root),threadpool_limits(config["cpu_threads"]):
        _,train,validation,signature,runtime = load_development(paths,config)
        if read_json(result_dir/"run_manifest.json")["fingerprint"] != signature:
            raise ExperimentBlocked("Run manifest input/config mismatch")
        if name not in candidate_names(config):
            raise ExperimentBlocked("Candidate not declared before Validation")
        if result_path.exists():
            previous = read_json(result_path)
            if previous["fingerprint"]!=signature:
                raise ExperimentBlocked("Candidate checkpoint uses different inputs")
            if previous["status"]=="PASS":
                if sha256_file(paths.project_root/previous["artifact"])!=previous["artifact_sha256"]:
                    raise ExperimentBlocked("Completed candidate artifact changed")
                return {"candidate":name,"status":"PASS","reused_checkpoint":True}
        try:
            with MemoryMonitor() as monitor:
                record = _fit_candidate(name,paths,config,train,validation,signature,runtime)
            record["memory"] = monitor.result
        except (ExperimentBlocked,MemoryError,FileNotFoundError) as error:
            record = {"candidate":name,"status":"BLOCKED","reason":str(error) if isinstance(error,ExperimentBlocked) else type(error).__name__}
        record["fingerprint"] = signature
        _json(result_path,record)
        return {"candidate":name,"status":record["status"],"macro_f1":record.get("metrics",{}).get("macro_f1"),"reason":record.get("reason")}


def _fit_candidate(name,paths,config,train,validation,signature,runtime):
    model_dir,result_dir = directories(paths)
    y_train,y_validation = train.label.to_numpy(),validation.label.to_numpy()
    indices = _smoke_indices(validation,config)
    provenance = {}
    if name.startswith("B_svd_"):
        family,components = "svd",int(name.rsplit("_",1)[1])
        preparation = read_json(result_dir/"tfidf_preparation.json")
        if preparation["status"]!="PASS" or preparation["fingerprint"]!=signature:
            raise ExperimentBlocked("Sparse TF-IDF preparation was blocked or inconsistent")
        cache_dir = paths.cache/"experiments"/EXPERIMENT
        matrices = {}
        for split in ("train","validation"):
            path = cache_dir/f"{split}.npz"
            if sha256_file(path)!=preparation[split]["sha256"]:
                raise ExperimentBlocked("TF-IDF cache checksum mismatch")
            matrices[split] = sparse.load_npz(path)
            if not sparse.issparse(matrices[split]):
                raise ExperimentBlocked("TF-IDF became dense before SVD")
        x_train,x_validation = matrices["train"],matrices["validation"]
        features = joblib.load(paths.project_root/runtime["artifact"],mmap_mode="r")[:-1]
        provenance = {"feature_source":runtime["artifact"],"feature_sha256":runtime["artifact_sha256"],"sparse_input":True}
    else:
        matrices,provenance = verified_embeddings(paths,config,train,validation,signature)
        x_train,x_validation = matrices["train"],matrices["validation"]
        family = "control" if name=="C_embedding_384_control" else "pca" if name.startswith("C_pca_") else "lda"
        components = int(name.rsplit("_",1)[1]) if family=="pca" else 384 if family=="control" else 2
    fit_seconds = None
    reducer_seconds = 0
    transform_seconds = 0
    variance = None
    energy_loss = None
    gate = None
    if family=="control":
        previous = read_json(paths.results/"experiments/multilingual_dl_01/validation_metrics.json")
        prior = next(c for c in previous["candidates"] if c["candidate"]=="B_minilm_svc" and c["status"]=="PASS")
        artifact_path = paths.project_root/prior["artifact"]
        if sha256_file(artifact_path)!=prior["artifact_sha256"]:
            raise ExperimentBlocked("Embedding control artifact checksum mismatch")
        artifact = joblib.load(artifact_path,mmap_mode="r")
        head = artifact.classifier
        z_validation = x_validation
    else:
        gate = require_ram(workspace_bytes(family,len(train),x_train.shape[1],components,len(validation)),config)
        reducer = make_reducer(family,components,config)
        z_train,z_validation,reducer_seconds,transform_seconds = fit_projection(reducer,x_train,x_validation,y_train)
        if family in {"svd","pca"}:
            variance = float(np.sum(reducer.explained_variance_ratio_))
            energy_loss = class_energy(x_validation,z_validation,y_validation,getattr(reducer,"mean_",None))
        if name=="D_lda_classifier":
            artifact_head = reducer
            head = reducer
            # LDA.predict consumes original 384D, not its 2D transformed matrix.
            z_validation = x_validation
            classifier_seconds = 0
            lda_figure(paths,config,z_train,reducer.transform(x_validation),y_train,y_validation)
        else:
            head = LinearSVC(random_state=config["seed"],**config["svc"])
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                head.fit(z_train,y_train)
            classifier_seconds = time.perf_counter()-start
            artifact_head = Pipeline([("reduce_train_only",reducer),("classifier",head)])
        fit_seconds = reducer_seconds+classifier_seconds
        artifact_path = model_dir/f"{name}.joblib"
        if family=="svd":
            artifact = Pipeline(features.steps+artifact_head.steps)
        else:
            artifact = EmbeddingClassifierArtifact(name,artifact_head,config["encoder"]["model_id"],config["encoder"]["revision"],
                config["encoder"]["max_sequence_length"],config["encoder"]["dimension"],list(head.classes_),"dimensionality_02_development_only")
        del z_train
    start = time.perf_counter()
    prediction = np.asarray(head.predict(z_validation))
    predict_seconds = time.perf_counter()-start
    if family!="control":
        loaded = save_artifact(artifact_path,artifact)
    else:
        loaded = artifact
    sample = validation.iloc[indices][["subject","body"]] if family=="svd" else np.asarray(x_validation[indices])
    predict = loaded.predict if family=="svd" else loaded.predict_embeddings
    np.testing.assert_array_equal(predict(sample),prediction[indices])
    timing = latency(predict,sample,config["benchmark_repeats"],"text_pipeline" if family=="svd" else "cached_embedding_pipeline")
    metrics = evaluate(validation,prediction)
    if family=="control" and not np.isclose(metrics["macro_f1"],prior["metrics"]["macro_f1"],atol=1e-12,rtol=0):
        raise ExperimentBlocked("Embedding control differs from saved Validation")
    reference_path = result_dir/"baseline_validation_predictions.npy"
    if sha256_file(reference_path)!=read_json(result_dir/"A_baseline.json")["validation_predictions_sha256"]:
        raise ExperimentBlocked("Baseline prediction checkpoint changed")
    reference = np.load(reference_path,allow_pickle=False)
    return {"candidate":name,"status":"PASS","dimensions":components,"family":family,"metrics":metrics,
        "fit_seconds":fit_seconds,"reducer_fit_transform_seconds":reducer_seconds,
        "classifier_fit_seconds":classifier_seconds if family!="control" else None,
        "validation_transform_seconds":transform_seconds,"validation_predict_seconds":predict_seconds,"latency":timing,
        "train_explained_variance_ratio_sum":variance,"validation_energy_lost_by_class":energy_loss,
        "variance_note":"LDA discriminant variance is not PCA/SVD total variance" if family=="lda" else None,
        "paired_errors_vs_baseline":paired_errors(y_validation,reference,prediction),
        "resource_gate":gate,"provenance":provenance,"calibrated_confidence_available":False,
        "artifact":artifact_path.relative_to(paths.project_root).as_posix(),"artifact_sha256":sha256_file(artifact_path),
        "artifact_bytes":artifact_path.stat().st_size,"reload_prediction_match":True,
        "warnings":[type(w.message).__name__ for w in caught] if family!="control" and name!="D_lda_classifier" else []}


def summarize(paths=None):
    paths = paths or resolve_v21_paths()
    config = load_config(paths)
    model_dir,result_dir = directories(paths)
    signature,_ = fingerprint(paths)
    manifest = read_json(result_dir/"run_manifest.json")
    if manifest["fingerprint"]!=signature:
        raise ExperimentBlocked("Summary configuration/input signature changed")
    records = []
    for name in ["A_baseline"]+candidate_names(config):
        path = result_dir/f"{name}.json"
        record = read_json(path) if path.exists() else {"candidate":name,"status":"BLOCKED","reason":"Candidate not run"}
        if record.get("fingerprint",signature)!=signature:
            raise ExperimentBlocked("Candidate summary fingerprint mismatch")
        if record["status"]=="PASS":
            artifact = paths.project_root/record["artifact"]
            if sha256_file(artifact)!=record["artifact_sha256"]:
                raise ExperimentBlocked("Artifact checksum differs before summary")
            pd.DataFrame(record["metrics"]["confusion_matrix"],index=CLASSES,columns=CLASSES).to_csv(result_dir/f"{name}_confusion_matrix.csv")
        records.append(record)
    baseline = records[0]["metrics"]["macro_f1"]
    reduced = [r for r in records if r["status"]=="PASS" and r.get("family") in {"svd","pca","lda"}]
    best = max(reduced,key=lambda r:r["metrics"]["macro_f1"]) if reduced else None
    output = {"experiment":EXPERIMENT,"scope":config["scope"],"manifest":manifest,"candidates":records,
        "best_reduced_candidate":best["candidate"] if best else None,
        "all_completed_reduced_candidates_below_baseline":all(r["metrics"]["macro_f1"]<baseline for r in reduced) if reduced else None,
        "runtime_changed":False,"heldout_opened":False,
        "source_sha256":{name:sha256_file(paths.project_root/name) for name in (
            "src/vietmailguard/v21/dimensionality_experiment.py","scripts/v21/dimensionality_reduction_02.py",
            "src/vietmailguard/v21/development_training.py","src/vietmailguard/v21/multilingual_dl_experiment.py",
            "src/vietmailguard/modeling.py","src/vietmailguard/embedding_modeling.py")}}
    _json(result_dir/"validation_metrics.json",output)
    rows = []
    for record in records:
        metrics = record.get("metrics",{})
        rows.append({"candidate":record["candidate"],"status":record["status"],"dimensions":record.get("dimensions"),
            **{key:metrics.get(key) for key in ("accuracy","macro_f1","phishing_recall","normal_to_spam_or_phishing")},
            "train_explained_variance":record.get("train_explained_variance_ratio_sum"),"fit_seconds":record.get("fit_seconds"),
            "latency_ms_per_email":record.get("latency",{}).get("median_ms_per_email"),
            "latency_scope":record.get("latency",{}).get("scope"),
            "sampled_peak_rss_gib":record.get("memory",{}).get("sampled_peak_rss_gib"),
            "artifact_bytes":record.get("artifact_bytes")})
    pd.DataFrame(rows).to_csv(result_dir/"model_comparison.csv",index=False)
    render_report(output,paths)
    files = list(model_dir.glob("*.joblib"))+[p for p in result_dir.iterdir() if p.is_file() and p.name!="artifact_manifest.json"]
    _json(result_dir/"artifact_manifest.json",{"fingerprint":signature,"files":[
        {"path":path.relative_to(paths.project_root).as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)}
        for path in sorted(files)]})
    return output


def render_report(output,paths):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"
    records = output["candidates"]
    manifest = output["manifest"]
    lines = ["# Dimensionality Reduction Experiment 02","","## Phạm vi và protocol","",
        "Thí nghiệm Development Validation, seed 42. Không thay V2 Production, V2.1 Preview hoặc LinearSVC runtime.",
        f"Train {manifest['train_rows']:,}; Validation {manifest['validation_rows']:,}; giữ nguyên assignment và SHA-256 bootstrap.",
        "Validation này từng phục vụ phát triển V2, không phải Final Test độc lập. Không mở held-out Test, không dùng TREC/synthetic Chinese, không gọi Gemini hoặc tải dependency/model.",
        "Đã kiểm tra lại exact/content, normalized, template, group và parent/translation bằng validator development hiện có.",
        "```json",json.dumps(manifest["leakage"],indent=2),"```","",
        "| Split | Language | Data origin | Class | Rows |","|---|---|---|---|---:|"]
    for split,items in manifest["distribution"].items():
        for item in items:
            lines.append(f"| {split} | {item['language']} | {item['data_origin']} | {item['label']} | {item['rows']} |")
    lines += ["","## Pipeline","",
        "- A: frozen TF-IDF Word+Character + balanced LinearSVC, load read-only; không refit TF-IDF hoặc classifier.",
        "- B: cùng TF-IDF fitted trên Train → TruncatedSVD randomized (100/200/300) → balanced LinearSVC C=1. TF-IDF luôn CSR sparse; chỉ latent output được dense.",
        "- C: cache MiniLM L2 384D, revision cố định → PCA (50/100/200, whiten=False) → balanced LinearSVC C=1. Có đối chứng MiniLM 384D + LinearSVC từ Experiment 01, không fit lại.",
        "- D: LDA solver=svd, tối đa 2 chiều với 3 lớp. So sánh LDA classifier và LDA 2D → balanced LinearSVC. Biểu đồ dùng trục LDA học từ Train, áp dụng nguyên trạng lên Validation.",
        "D_lda_classifier vẫn nhận input 384D và dùng predict của LDA; n_components=2 chỉ giới hạn transform. D_lda2_svc mới dùng tọa độ 2D làm input LinearSVC. Không coi hai cách dự đoán là cùng một classifier.",
        "Tất cả reducer/scaler/classifier mới chỉ fit trên Train. LDA chỉ nhận y_train, không nhận nhãn Validation khi fit. PCA không whitening hoặc refit trên Validation; encoder không fine-tune.",
        "Chọn LinearSVC cho PCA để tập trung vào ảnh hưởng phép giảm chiều và tiết kiệm tài nguyên; không sweep MLP/hyperparameter ngoài cấu hình đã khai báo.",
        "Mọi phép fit dùng CPU và thread limit trong config. Cache embeddings được kiểm tra ordered content keys, model/revision, finite/L2/shape/dtype và SHA-256 đối chiếu receipt encoder đã xác minh ở Experiment 01; không nạp encoder mới.",
        "","## So sánh Validation","",
        "| Candidate | Status | Dim | Macro F1 | Phishing Recall | Normal→Spam/Phishing | Variance* | Fit s | ms/email** | RSS GiB*** | Artifact MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        metrics = record.get("metrics",{})
        numbers = (record.get("dimensions"),metrics.get("macro_f1"),metrics.get("phishing_recall"),
            metrics.get("normal_to_spam_or_phishing"),record.get("train_explained_variance_ratio_sum"),record.get("fit_seconds"),
            record.get("latency",{}).get("median_ms_per_email"),record.get("memory",{}).get("sampled_peak_rss_gib"),
            record["artifact_bytes"]/2**20 if "artifact_bytes" in record else None)
        lines.append(f"| {record['candidate']} | {record['status']} | "+" | ".join(fmt(v) for v in numbers)+" |")
        if record.get("reason"):
            lines += ["",f"{record['candidate']}: {record['reason']}"]
    lines += ["","*Variance là tổng explained_variance_ratio_ trên Train của PCA/SVD; không đồng nghĩa phần thông tin phân loại được bảo toàn. LDA discriminant variance là đại lượng khác nên không điền chung cột này.",
        "**Median 3 lần sau warm-up, cùng batch tối đa 8 EN + 8 VI. A/B gồm compose+TF-IDF+projection/classifier; C/D và embedding control chỉ gồm pipeline trên embeddings đã cache, không gồm encoder. Không so trực tiếp hai scope này như độ trễ end-to-end tương đương.",
        "***RSS peak lấy mẫu 100 ms, gồm resident input/cache và quá trình reload/inference; không phải phép đo allocation riêng reducer hay peak tuyệt đối. Native process peak và RAM trước/sau lưu trong JSON. Artifact B chứa cả fitted TF-IDF; C/D cần encoder local đúng revision ngoài artifact head.",
        "RSS của D_lda_classifier còn bao gồm dựng hình bằng Matplotlib; không quy toàn bộ phần chênh lệch đó cho thuật toán LDA.",
        "Fit s gồm reducer.fit_transform(Train) + classifier.fit; không gồm tạo TF-IDF/embeddings. Baseline/control được tái sử dụng nên fit N/A."]
    for record in records:
        if record["status"]!="PASS":
            continue
        metrics = record["metrics"]
        lines += ["",f"### {record['candidate']}","",f"Accuracy: {fmt(metrics['accuracy'])}.",
            "","| Class | Precision | Recall | F1 | Support |","|---|---:|---:|---:|---:|"]
        for label,values in metrics["per_class"].items():
            lines.append(f"| {label} | {fmt(values['precision'])} | {fmt(values['recall'])} | {fmt(values['f1'])} | {values['support']} |")
        lines += ["","Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.","```json",json.dumps(metrics["confusion_matrix"]),"```",
            "","| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |","|---|---:|---:|---:|"]
        for language,values in metrics["by_language"].items():
            lines.append(f"| {language} | {values['rows']} | {fmt(values['macro_f1'])} | {fmt(values['phishing_recall'])} |")
        if record.get("validation_energy_lost_by_class"):
            lines += ["","Validation squared-energy lost theo lớp (PCA: tâm là mean Train):", "```json",
                json.dumps(record["validation_energy_lost_by_class"],indent=2),"```"]
        if record.get("paired_errors_vs_baseline"):
            lines += ["","So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:","```json",json.dumps(record["paired_errors_vs_baseline"],indent=2),"```"]
        if record.get("warnings"):
            lines += ["",f"Fit warnings: {', '.join(record['warnings'])}."]
    _,result_dir = directories(paths)
    if (result_dir/"lda_projection.png").exists():
        lines += ["","## Trực quan LDA","", "![LDA Train và Development Validation](results/v21_multilingual/experiments/dimensionality_reduction_02/lda_projection.png)",
            "Lấy mẫu phân tầng cố định tối đa 1.000 điểm/lớp/panel chỉ để dễ nhìn. Không thay đổi dữ liệu đánh giá. Trục dùng nhãn Train nên hình Train có thể tách đẹp hơn; không coi biểu đồ là bằng chứng hiệu năng độc lập."]
    lines += ["","## Kết luận","",f"Baseline thực tế: {records[0]['metrics']['macro_f1']:.6f}; reduced candidate tốt nhất: {output['best_reduced_candidate']}."]
    if output["all_completed_reduced_candidates_below_baseline"]:
        lines += ["Mọi candidate giảm chiều đã hoàn tất đều kém baseline về Macro F1 Development. Giữ nguyên model chính; không thay runtime."]
    control = next((r for r in records if r['candidate']=='C_embedding_384_control' and r['status']=='PASS'),None)
    lines += ["","| Phép giảm chiều | Δ Macro F1 so với control cùng representation | Δ Phishing Recall |", "|---|---:|---:|"]
    for record in records:
        if record['status']!='PASS' or record.get('family') not in {'svd','pca','lda'}:
            continue
        reference = records[0] if record['family']=='svd' else control
        if reference:
            lines.append(f"| {record['candidate']} | {record['metrics']['macro_f1']-reference['metrics']['macro_f1']:+.6f} | {record['metrics']['phishing_recall']-reference['metrics']['phishing_recall']:+.6f} |")
    lines += ["Giảm chiều có thể loại các hướng ít variance nhưng chứa tín hiệu hiếm của Spam/Phishing. Bảng per-class, energy loss và số baseline-correct→candidate-wrong cho biết trade-off quan sát được; không đủ để khẳng định một từ khóa cụ thể bị mất là nguyên nhân.",
        "PCA center theo mean Train và giữ phương sai tổng thể; TruncatedSVD không center sparse TF-IDF, xấp xỉ ma trận gốc; LDA là phép supervised dùng nhãn Train. Cả ba mục tiêu đều không bảo đảm Macro F1/Phishing Recall tốt hơn classifier gốc.",
        "Baseline tuyến tính ba lớp chỉ cần sparse dot với các hướng phân loại. Thêm SVD 100–300 chiều phải tính thêm sparse×dense projection và lưu dense components; số chiều đầu vào classifier giảm không bảo đảm artifact nhỏ hơn hay inference nhanh hơn. PCA cũng thêm phép chiếu; mọi lợi ích trên head còn phải tính chi phí MiniLM encoder khi triển khai thực tế.",
        "So sánh PCA với control 384D trong cùng họ embeddings, SVD với TF-IDF cùng vectorizer. Khác biệt TF-IDF/MiniLM còn bao gồm encoder và truncation 128 tokens; không quy mọi thay đổi cho giảm chiều.",
        "VI Validation chỉ 172 normal + 8 spam, 0 phishing; ZH không có hàng. Không có native Vietnamese/Chinese three-class benchmark độc lập. Không tuyên bố Phishing VI/ZH, Production hoặc Final Evaluation PASS.",
        "","## Tái lập","","```powershell",
        ".\\.venv\\Scripts\\python.exe scripts/v21/dimensionality_reduction_02.py --preflight",
        ".\\.venv\\Scripts\\python.exe scripts/v21/dimensionality_reduction_02.py --run","```",
        "Mỗi candidate chạy process riêng để giải phóng workspace. Stage hoàn tất có hash/checkpoint được tái sử dụng; artifact hiện có không bị ghi đè. Candidate thiếu RAM/cache ghi BLOCKED, không tự hạ n_components hoặc đổi model.",
        "Config: config/v21/experiments/dimensionality_reduction_02.json. Models/results/cache lưu riêng dưới namespace experiments/dimensionality_reduction_02. Không chứa raw email trong report/plot."]
    verification_path = result_dir/'verification_receipt.json'
    if verification_path.exists():
        verification = read_json(verification_path)
        lines += ["","## Kiểm thử, bảo toàn và file tạo mới","","```json",json.dumps(verification,ensure_ascii=False,indent=2),"```",
            "Source/config/test mới: config/v21/experiments/dimensionality_reduction_02.json; src/vietmailguard/v21/dimensionality_experiment.py; scripts/v21/dimensionality_reduction_02.py; tests/test_v21_dimensionality_experiment.py; DIMENSIONALITY_REDUCTION_EXPERIMENT_02.md.",
            "Các harness/receipt local nằm trong .local-test-deps/dimensionality-02/ (Git ignored). Không sửa source/UI/runtime cũ hoặc các file Experiment 01. Không commit/push."]
    (paths.project_root/"DIMENSIONALITY_REDUCTION_EXPERIMENT_02.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
