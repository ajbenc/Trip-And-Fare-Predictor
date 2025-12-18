param([switch]$WhatIf)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

$targets = @(
  'app_streamlit.py',
  'add_may_and_retrain.py',
  'add_may_step1_clean.py',
  'add_may_step2_features.py',
  'add_may_step3_split.py',
  'add_may_step4_train.py',
  'analysis_retraining_vs_calibration.py',
  'analyze_duration_predictions.py',
  'compare_68_vs_56.py',
  'engineer_68_features_full.py',
  'model_vs_calibration_breakdown.py',
  'simulation_retraining_impact.py',
  'split_68_features.py',
  'train_68_features.py',
  'why_not_retrain_analysis.py'
)

$deleted = @()
$missing = @()
foreach ($t in $targets) {
  $p = Join-Path $root $t
  if (Test-Path $p) {
    if ($WhatIf) {
      Write-Host "Would remove $p"
    } else {
      Remove-Item -LiteralPath $p -Force
      Write-Host "Removed $p"
      $deleted += $p
    }
  } else {
    $missing += $p
  }
}

Write-Host "--- Summary ---"
Write-Host ("Deleted: {0}" -f $deleted.Count)
Write-Host ("Missing: {0}" -f $missing.Count)
if ($missing.Count -gt 0) {
  $missing | ForEach-Object { Write-Host "Missing: $_" }
}
