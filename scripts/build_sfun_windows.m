%% build_sfun_windows.m
% Compiles sfun_rom_lstm_qat.c for the current MATLAB platform.
% Run once on Windows to generate sfun_rom_lstm_qat.mexw64 in src/.
% Works on any platform: Windows (.mexw64), Mac ARM (.mexmaca64), Mac Intel (.mexmaci64), Linux (.mexa64)
%
% Usage: run this script from the project root directory, or set PROJ below.

PROJ = '/path/to/Claude_ROM_session4';   % <-- update if needed, or run from project root
if isfolder(fullfile(pwd, 'src'))
    PROJ = pwd;   % auto-detect if running from project root
end

SRC = fullfile(PROJ, 'src');
addpath(SRC);

src_file = fullfile(SRC, 'sfun_rom_lstm_qat.c');
assert(isfile(src_file), 'Cannot find %s', src_file);

fprintf('Compiling sfun_rom_lstm_qat for %s...\n', mexext);
old = cd(SRC);
try
    mex('-O', ['-I' SRC], 'sfun_rom_lstm_qat.c');
    fprintf('Success: %s\n', fullfile(SRC, ['sfun_rom_lstm_qat.' mexext]));
catch ME
    cd(old);
    error('Compilation failed: %s\n\nTip: run "mex -setup C" to configure your C compiler first.', ME.message);
end
cd(old);
