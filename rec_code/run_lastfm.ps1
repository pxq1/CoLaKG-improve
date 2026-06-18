param(
    [int]$Epochs = 1000,
    [string]$EnvName = "CoLaKG",
    [string]$CudaDevice = "0"
)

Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = $CudaDevice

$argsList = @(
    "main.py",
    "--epochs=$Epochs",
    "--dataset=lastfm",
    "--bpr_batch=1024",
    "--decay=1e-4",
    "--lr=0.001",
    "--layer=3",
    "--seed=2020",
    "--topks=[10,20]",
    "--recdim=64",
    "--use_drop_edge=0",
    "--keepprob=1.0",
    "--neighbor_k=10",
    "--dropout_i=0.6",
    "--dropout_u=0.2",
    "--dropout_n=0.6",
    "--item_semantic_emb_file=../data/lastfm/lastfm_embeddings_simcse_kg.pt",
    "--user_semantic_emb_file=../data/lastfm/lastfm_embeddings_simcse_kg_user.pt"
)

conda run --no-capture-output -n $EnvName python -u @argsList
