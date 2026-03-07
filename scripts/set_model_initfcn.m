%% set_model_initfcn.m
% Sets model callbacks so both models work when opened cold from the GUI.
%
%  PreLoadFcn  – fires BEFORE block parameters are evaluated at model load.
%                Creates default throttle_ts / sa_ts in base workspace.
%  InitFcn     – fires at simulation start; adds src/ to path for the MEX.
%
% Strings built with single-quoted char arrays + char(10) for newlines
% to avoid the set_param double-quote / newline truncation bug.

PROJ = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
addpath(fullfile(PROJ, 'src'));

% ---- PreLoadFcn ----------------------------------------------------------
% Target code that must execute:
%   if ~evalin('base','exist(''throttle_ts'',''var'')')
%       t_def=(0:0.05:25)';
%       assignin('base','throttle_ts',[t_def,30*ones(size(t_def))]);
%       assignin('base','sa_ts',[t_def,15*ones(size(t_def))]);
%   end
%
% Each '' in a single-quoted MATLAB string literal becomes one '.
% To produce ''throttle_ts'' in the stored string, write ''''throttle_ts'''' here.

preLoadFcn = [ ...
    'if ~evalin(''base'',''exist(''''throttle_ts'''',''''var'''')'')' char(10) ...
    '    t_def=(0:0.05:25)'';' char(10) ...
    '    assignin(''base'',''throttle_ts'',[t_def,30*ones(size(t_def))]);' char(10) ...
    '    assignin(''base'',''sa_ts'',[t_def,15*ones(size(t_def))]);' char(10) ...
    '    disp(''[ROM model] Default inputs: throttle=30 deg, SA=15 deg.'');' char(10) ...
    'end'];

% ---- InitFcn -------------------------------------------------------------
% Adds src/ to path so sfun_rom_lstm_qat.mexmaca64 is found at sim start.
initFcn = 'addpath(fullfile(fileparts(which(bdroot)),''src''));';

% ---- Apply to both models -----------------------------------------------
models = {'enginespeed_qat_sfun', 'enginespeed_hifi_val'};

for mi = 1:numel(models)
    m = models{mi};
    f = fullfile(PROJ, [m '.slx']);
    if ~isfile(f), fprintf('[SKIP] %s\n', f); continue; end
    if bdIsLoaded(m), close_system(m, 0); end
    load_system(f);
    set_param(m, 'PreLoadFcn', preLoadFcn);
    set_param(m, 'InitFcn',    initFcn);
    % Verify stored correctly
    stored = get_param(m, 'PreLoadFcn');
    if contains(stored, 'throttle_ts') && contains(stored, char(10))
        fprintf('[OK]  %s  (PreLoadFcn has %d lines)\n', m, sum(stored==char(10))+1);
    else
        fprintf('[WARN] %s  – PreLoadFcn may be malformed\n', m);
        disp(stored);
    end
    save_system(m, f);
    close_system(m, 0);
end
fprintf('\nDone.\n');
