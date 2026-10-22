# Treina os 6 modelos da comparação divisão aleatória x divisão por série, um depois do outro:
# modelo inicial (4 blocos, parada antecipada) e modelo final (6 blocos + balanço de branco + cutout, 40 épocas),
# com as sementes 42, 1 e 2, na pasta dataset/split_aleatoria. Usa os scripts de treino existentes, sem alterá-los.
# Cada treino que já terminou (com resultados.json) é pulado pelo próprio script, então dá para rodar de novo.
# Rodar a partir da raiz do projeto:  powershell -ExecutionPolicy Bypass -File rodar_divisao_aleatoria.ps1

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:PYTHONPATH = "."
$P = "dataset/split_aleatoria"
$E = "artifacts/experiments"
Start-Transcript -Path "$E/log_divisao_aleatoria.txt" -Append

foreach ($s in 42, 1, 2) {
    Write-Host "`n===== Modelo inicial, semente $s =====" -ForegroundColor Cyan
    python evolucao_modelo/train_group_split2.py --pasta $P --saida "$E/aleatoria_inicial_s$s" --semente $s
}
foreach ($s in 42, 1, 2) {
    Write-Host "`n===== Modelo final, semente $s =====" -ForegroundColor Cyan
    python train_group_split_profundo_balanco_branco_cutout2.py --pasta $P --saida "$E/aleatoria_final_s$s" --semente $s --sem_parada
}

Write-Host "`n===== Resumo =====" -ForegroundColor Cyan
python evolucao_modelo/resumo_divisao_aleatoria.py
Stop-Transcript
