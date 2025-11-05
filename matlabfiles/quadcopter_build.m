%% =======================================================================
%  BUILD SCRIPT for Quadcopter Model — MATLAB Coder
%  Generates  a DLL (on Windows)
%  Author: Solal Baudoin
%  =======================================================================

clc;
clear;
disp('=== Building quadcopter_model ===');

%% 1. Check function visibility
if exist('quadcopter_model.m', 'file') ~= 2
    error('❌ Function "quadcopter_model.m" not found in current directory.');
end

%% 2. Check compiler setup
try
    mexCompiler = mex.getCompilerConfigurations('C', 'Selected');
    if isempty(mexCompiler)
        disp('⚙️ No C compiler configured. Launching mex -setup ...');
        mex -setup
    else
        fprintf('✅ Using compiler: %s\n', mexCompiler.Name);
    end
catch
    disp('⚙️ Could not detect compiler. Running setup...');
    mex -setup
end

%% 3. Configure code generation
cfg = coder.config('lib');          % 'lib' = portable C library
cfg.TargetLang = 'C';               % or 'C++'
cfg.GenerateReport = true;          % HTML report
cfg.GenerateExampleMain = 'DoNotGenerate'; % optional

disp('✅ Configuration ready');

%% 4. Define argument types
ARGS = {zeros(12,1), zeros(4,1)};  % State vector, control input

%% 5. Launch code generation
disp('🚀 Launching MATLAB Coder ...');
codegen('quadcopter_model', '-config', cfg, '-args', ARGS);

%% 6. Locate result folder automatically
outDir = fullfile(pwd, 'codegen', 'lib', 'quadcopter_model');
if ~isfolder(outDir)
    warning('⚠️ Could not locate expected output folder.');
else
    disp(['✅ Code generation complete: ', outDir]);
end

%% 7. Open report automatically
reportFile = fullfile(outDir, 'html', 'index.html');
if isfile(reportFile)
    web(reportFile, '-browser');
end

%% 8. (Optional) Build DLL automatically on Windows
if ispc
    disp('🧱 Attempting to build DLL with MSVC...');
    cd(outDir);

    % Récupérer tous les .c générés
    cFiles = dir('*.c');
    if isempty(cFiles)
        error('No C files found in %s', outDir);
    end

    % Construire la ligne de commande CL
    srcFiles = strjoin({cFiles.name}, ' ');
    cmd = sprintf('cl /LD /O2 %s', srcFiles);
    [status, cmdout] = system(cmd);

    cd(fullfile(pwd, '..', '..', '..'));  % revenir au dossier projet

    if status == 0
        disp('✅ DLL successfully built: codegen/lib/quadcopter_model/quadcopter_model.dll');
    else
        disp('⚠️ DLL build failed (check Visual Studio setup).');
        disp(cmdout);
    end
end

disp('=== Done ===');
