@echo off
rem Initialise a native build environment for the optional CUDA/C++ extension.
rem Nothing here is machine-specific: Visual Studio is found through vswhere and
rem the CUDA toolkit is taken from CUDA_HOME or the newest installed version.
rem
rem   scripts\native_env.bat python benchmarks\validate_native.py
rem
setlocal

rem --- CUDA toolkit -----------------------------------------------------------
if not defined CUDA_HOME (
  for %%R in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA") do (
    if exist "%%~R" (
      for /f "delims=" %%V in ('dir /b /ad /o-n "%%~R" 2^>nul') do (
        if not defined CUDA_HOME set "CUDA_HOME=%%~R\%%V"
      )
    )
  )
)
if defined CUDA_HOME set "PATH=%CUDA_HOME%\bin;%PATH%"

rem --- MSVC build tools -------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto run
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%I"
if defined VSDIR if exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

:run
if not defined CUDA_HOME echo [native_env] CUDA_HOME not found; the native extension will not compile. 1>&2
%*
