# Where to install aria2
$installDir = "$env:USERPROFILE\aria2"
$aria2Exe   = "$installDir\aria2c.exe"

# === Choose what to download ===
# "single"  -> downloads the single file at $datasetUrl (e.g. puzzles.csv)
# "shards"  -> downloads all action_value shards 00000..02147
$downloadMode = "shards"   # <-- set to "single" or "shards"

# Single-file example (kept from your script)
#$datasetUrl = "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"

# Output directory (created if missing)
$outDir = "H:\ChessData" #"$PWD"    # or set a custom folder, e.g. "$env:USERPROFILE\Downloads\searchless_chess"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

# Download aria2 portable build if not installed
if (-not (Test-Path $aria2Exe)) {
    Write-Host "Downloading aria2 portable build..."
    $zipUrl  = "https://github.com/aria2/aria2/releases/download/release-1.36.0/aria2-1.36.0-win-64bit-build1.zip"
    $zipPath = "$env:TEMP\aria2.zip"

    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force

    # Move aria2c.exe to installDir root for convenience
    $exeFound = Get-ChildItem -Path $installDir -Recurse -Filter "aria2c.exe" | Select-Object -First 1
    if ($exeFound) {
        Copy-Item $exeFound.FullName $aria2Exe -Force
    }

    Remove-Item $zipPath -Force
}

# === Download logic ===
if ($downloadMode -eq "single") {
    Write-Host "Starting download of $datasetUrl ..."
    & $aria2Exe -c -x 16 -s 16 -d $outDir $datasetUrl

} elseif ($downloadMode -eq "shards") {
    Write-Host "Starting download of action_value shards (00000..02147) ..."

    # Build a temp URL list (faster for aria2 than invoking per-file)
    $tmpList = Join-Path $env:TEMP "action_value_urls.txt"
    if (Test-Path $tmpList) { Remove-Item $tmpList -Force }

    for ($i = 0; $i -le 2147; $i++) {
        $idx = "{0:D5}" -f $i
        $url = "https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag"
        Add-Content -Path $tmpList -Value $url
    }

    # Use aria2 in batch mode with resume, parallel connections, and output dir
    & $aria2Exe -c -x 16 -s 16 -d $outDir -i $tmpList

    # Optional: Remove the list after download
    # Remove-Item $tmpList -Force
} else {
    throw "Unknown downloadMode: $downloadMode (expected 'single' or 'shards')"
}
