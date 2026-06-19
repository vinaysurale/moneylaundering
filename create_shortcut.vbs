Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "C:\Users\vinay\Desktop\AML Dashboard.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "c:\project 3\run_aml_dashboard.bat"
oLink.WorkingDirectory = "c:\project 3"
oLink.Save
WScript.Echo "Shortcut created successfully"
