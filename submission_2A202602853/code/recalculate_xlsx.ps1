param([switch]$Render, [string]$WorkbookPath = (Join-Path $PSScriptRoot '../experiments.xlsx'))
$ErrorActionPreference = 'Stop'
$workbookTaskPath = [IO.Path]::GetFullPath($WorkbookPath)
$excelTask = $null
$workbookTask = $null
try {
    $excelTask = New-Object -ComObject Excel.Application
    $excelTask.Visible = $false
    $excelTask.DisplayAlerts = $false
    $workbookTask = $excelTask.Workbooks.Open($workbookTaskPath)
    $excelTask.CalculateFullRebuild()
    foreach ($sheetTask in $workbookTask.Worksheets) {
        if ($sheetTask.Name -in @('Experiments', 'Summary')) { $sheetTask.UsedRange.Rows.AutoFit() | Out-Null }
    }
    $workbookTask.Save()
    if ($Render) {
        $renderTaskDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.verification/part34'))
        New-Item -ItemType Directory -Force -Path $renderTaskDir | Out-Null
        foreach ($sheetTask in $workbookTask.Worksheets) {
            $sheetTask.PageSetup.Orientation = 2
            $sheetTask.PageSetup.Zoom = $false
            $sheetTask.PageSetup.FitToPagesWide = 1
            $sheetTask.PageSetup.FitToPagesTall = 1
            $sheetTask.PageSetup.PaperSize = 8
            $sheetTask.PageSetup.PrintArea = switch ($sheetTask.Name) {
                'Legend' { 'A1:F56' }
                'Experiments' { 'A1:AG23' }
                'Seeds' { 'A1:D14' }
                'Summary' { 'A1:H14' }
            }
            $pdfTaskPath = Join-Path $renderTaskDir ($sheetTask.Name + '.pdf')
            $sheetTask.ExportAsFixedFormat(0, $pdfTaskPath)
        }
    }
    $workbookTask.Close($false)
    $workbookTask = $null
    Write-Output "Recalculated and saved: $workbookTaskPath"
} finally {
    if ($null -ne $workbookTask) { $workbookTask.Close($false) }
    if ($null -ne $excelTask) { $excelTask.Quit() }
}
