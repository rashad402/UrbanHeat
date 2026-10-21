# UrbanHeat Planner: runs the server with no console window and lives in the system tray.
#
#   start (default)  start the server hidden, open the browser when it answers, show a tray icon
#   stop             ask a running tray instance to shut the server down and exit
#
# Started by "UrbanHeat Planner.vbs" (which hides this script's own window). Because there is no
# console, nothing here can print: the server's output goes to data\planner.log, and this script's
# own failures go to data\planner_launcher.log.

param([ValidateSet("start", "stop")][string]$Action = "start")

$script:Root   = Split-Path -Parent $PSScriptRoot          # this file lives in <project>\launcher
$script:Port   = 8080
$script:Url    = "http://127.0.0.1:$($script:Port)"
$script:Icon   = Join-Path $PSScriptRoot "UrbanHeat.ico"
$script:Log    = Join-Path $script:Root "data\planner.log"
$script:Fail   = Join-Path $script:Root "data\planner_launcher.log"
$QuitName      = "UrbanHeat.Planner.Quit"
$MutexName     = "UrbanHeat.Planner.Tray"
$script:ReadyTimeoutS = 120

function Write-Fail($msg) {
    try {
        New-Item -ItemType Directory -Force (Split-Path $script:Fail) | Out-Null
        Add-Content -Path $script:Fail -Value ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
    } catch { }
}

function Test-Planner {
    try {
        Invoke-WebRequest "$($script:Url)/api/meta" -UseBasicParsing -TimeoutSec 1 | Out-Null
        return $true
    } catch { return $false }
}

# ---------------------------------------------------------------- stop
if ($Action -eq "stop") {
    $h = $null
    try { $h = [System.Threading.EventWaitHandle]::OpenExisting($QuitName) } catch { }
    if ($h) { [void]$h.Set(); $h.Dispose() }
    exit 0
}

# ---------------------------------------------------------------- start
try {
    # One tray per user. A second launch just opens the page.
    $created = $false
    $mutex = New-Object System.Threading.Mutex($true, $MutexName, [ref]$created)
    if (-not $created) { Start-Process $script:Url; exit 0 }

    # Something else is already serving the planner (for example run_planner.bat in a console).
    # Open it, but do not adopt a server this script did not start and could not cleanly stop.
    if (Test-Planner) { Start-Process $script:Url; $mutex.ReleaseMutex(); exit 0 }

    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $script:quit    = New-Object System.Threading.EventWaitHandle($false, [System.Threading.EventResetMode]::ManualReset, $QuitName)
    $script:server  = $null
    $script:ready   = $false
    $script:started = Get-Date
    $script:told    = $false      # have we already said the server stopped?

    function Start-Server {
        New-Item -ItemType Directory -Force (Split-Path $script:Log) | Out-Null
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName         = $env:ComSpec
        # cmd does the redirection, so the server's output lands in the log instead of nowhere.
        $psi.Arguments        = "/c python -m uvicorn api.planner:app --port $($script:Port) > `"$($script:Log)`" 2>&1"
        $psi.WorkingDirectory = $script:Root       # the app reads .env and configs relative to here
        $psi.UseShellExecute  = $false
        $psi.CreateNoWindow   = $true
        $script:server  = [System.Diagnostics.Process]::Start($psi)
        $script:ready   = $false
        $script:told    = $false
        $script:started = Get-Date
        $notify.Text = "UrbanHeat Planner: starting..."
    }

    function Stop-Server {
        if ($script:server -and -not $script:server.HasExited) {
            & taskkill.exe /PID $script:server.Id /T /F 2>&1 | Out-Null     # cmd, python and its children
        }
    }

    function Show-Balloon($text, $kind) {
        $notify.ShowBalloonTip(5000, "UrbanHeat Planner", $text, $kind)
    }

    function Exit-All {
        $timer.Stop()
        Stop-Server
        $notify.Visible = $false
        $notify.Dispose()
        $script:quit.Dispose()
        try { $mutex.ReleaseMutex() } catch { }
        [System.Windows.Forms.Application]::ExitThread()
    }

    # ---- tray icon and menu
    $notify = New-Object System.Windows.Forms.NotifyIcon
    $notify.Icon = New-Object System.Drawing.Icon($script:Icon)
    $notify.Text = "UrbanHeat Planner: starting..."
    $notify.Visible = $true

    $menu = New-Object System.Windows.Forms.ContextMenuStrip
    $miOpen    = $menu.Items.Add("Open planner")
    $miLog     = $menu.Items.Add("Show server log")
    $miRestart = $menu.Items.Add("Restart server")
    [void]$menu.Items.Add("-")
    $miQuit    = $menu.Items.Add("Quit (stop server)")
    $miOpen.Font = New-Object System.Drawing.Font($miOpen.Font, [System.Drawing.FontStyle]::Bold)
    $notify.ContextMenuStrip = $menu

    $miOpen.Add_Click({ Start-Process $script:Url })
    $notify.Add_DoubleClick({ Start-Process $script:Url })
    $miLog.Add_Click({ if (Test-Path $script:Log) { Start-Process notepad.exe $script:Log } })
    $miRestart.Add_Click({ Stop-Server; Start-Server })
    $miQuit.Add_Click({ Exit-All })

    # ---- one-second heartbeat: readiness, crashes, and the stop signal
    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 1000
    $timer.Add_Tick({
        if ($script:quit.WaitOne(0)) { Exit-All; return }

        if ($script:server -and $script:server.HasExited) {
            if (-not $script:told) {
                $script:told = $true
                $notify.Text = "UrbanHeat Planner: server stopped"
                Show-Balloon "The server stopped. Right-click this icon to see the log or restart it." ([System.Windows.Forms.ToolTipIcon]::Warning)
            }
            return
        }

        if (-not $script:ready) {
            if (Test-Planner) {
                $script:ready = $true
                $notify.Text = "UrbanHeat Planner: running"
                Start-Process $script:Url
                Show-Balloon "Running. Right-click this icon to quit." ([System.Windows.Forms.ToolTipIcon]::Info)
            } elseif (((Get-Date) - $script:started).TotalSeconds -gt $script:ReadyTimeoutS -and -not $script:told) {
                $script:told = $true
                $notify.Text = "UrbanHeat Planner: did not start"
                Show-Balloon "It is taking too long to start. Right-click this icon and choose Show server log." ([System.Windows.Forms.ToolTipIcon]::Error)
            }
        }
    })

    Start-Server
    $timer.Start()
    [System.Windows.Forms.Application]::Run((New-Object System.Windows.Forms.ApplicationContext))
}
catch {
    Write-Fail ($_ | Out-String)
    exit 1
}
