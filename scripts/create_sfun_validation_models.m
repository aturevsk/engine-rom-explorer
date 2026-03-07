%% create_sfun_validation_models.m
% ==========================================================================
% Builds two Simulink models for closed-loop QAT ROM validation:
%
%   enginespeed_hifi_val.slx
%     – Copy of enginespeed.slx with From Workspace inputs for
%       Throttle (throttle_ts) and Spark Advance (sa_ts).
%     – Signal logging on Speed and Torque.
%
%   enginespeed_qat_sfun.slx
%     – Combustion SubSystem REPLACED by Level-2 S-Function
%       (sfun_rom_lstm_qat.mexmaca64) + a Mux that bundles the 3
%       scalar inputs [AirCharge, Speed, SparkAdv] into the width-3
%       S-Function input port.
%     – S-Function maintains LSTM h/c state in DWork across time steps.
%     – Closed-loop: S-Function Torque output drives Vehicle Dynamics.
%
% R2025b API notes:
%   - delete_line BEFORE delete_block
%   - From Workspace: 'Interpolate','off', 'OutputAfterFinalValue',
%                     'Holding final value', 'OutDataTypeStr','double'
%   - Signal logging on PORT handle (not line handle)
%   - S-Function ports are resolved when FunctionName is set and the
%     .mexmaca64 file is on the MATLAB path (no ConfigSet custom code needed)
% ==========================================================================

clc;

%% 0. Paths ----------------------------------------------------------------
PROJ  = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
SRC   = fullfile(PROJ, 'src');
cd(PROJ);
if ~any(strcmp(SRC, strsplit(path, pathsep))), addpath(SRC); end

orig_mdl   = 'enginespeed';
hifi_mdl   = 'enginespeed_hifi_val';
sfun_mdl   = 'enginespeed_qat_sfun';
orig_file  = fullfile(PROJ, [orig_mdl  '.slx']);
hifi_file  = fullfile(PROJ, [hifi_mdl  '.slx']);
sfun_file  = fullfile(PROJ, [sfun_mdl  '.slx']);

assert(isfile(orig_file), 'Cannot find %s', orig_file);
assert(isfile(fullfile(SRC,'sfun_rom_lstm_qat.mexmaca64')), ...
    'MEX binary not found: %s/sfun_rom_lstm_qat.mexmaca64', SRC);

warning('off', 'Simulink:blocks:DelaySmallerThanOneSampleTime');
warning('off', 'Simulink:blocks:NegativeDelayValue');
warning('off', 'Simulink:Engine:UndefinedOutput');

% =========================================================================
%% PART A – Build enginespeed_hifi_val.slx
% =========================================================================
fprintf('\n=== Part A: Building %s ===\n', hifi_mdl);

if bdIsLoaded(hifi_mdl), close_system(hifi_mdl, 0); end
copyfile(orig_file, hifi_file, 'f');
load_system(hifi_file);

% ---- A1. Replace Throttle (degrees) Constant → From Workspace -----------
thr_path = [hifi_mdl '/Throttle (degrees)'];
tm_path  = [hifi_mdl '/Throttle & Manifold'];
thr_pos  = get_param(thr_path, 'Position');

tm_ph = get_param(tm_path, 'PortHandles');
delete_line_at_port(tm_ph.Inport(1));
delete_block(thr_path);

add_block('simulink/Sources/From Workspace', thr_path, ...
    'Position',              thr_pos,               ...
    'VariableName',          'throttle_ts',          ...
    'Interpolate',           'off',                  ...
    'OutputAfterFinalValue', 'Holding final value',  ...
    'OutDataTypeStr',        'double',               ...
    'SampleTime',            '0');

thr_ph = get_param(thr_path, 'PortHandles');
tm_ph  = get_param(tm_path,  'PortHandles');
add_line(hifi_mdl, thr_ph.Outport(1), tm_ph.Inport(1), 'autorouting','on');
fprintf('  Throttle (degrees) Constant → From Workspace (throttle_ts)\n');

% ---- A2. Replace Spark Advance Constant → From Workspace ----------------
sa_path    = [hifi_mdl '/Spark Advance'];
goto1_path = find_block_type(hifi_mdl, 'Goto', 1);
sa_pos     = get_param(sa_path, 'Position');

if ~isempty(goto1_path)
    goto1_ph = get_param(goto1_path, 'PortHandles');
    delete_line_at_port(goto1_ph.Inport(1));
end

delete_block(sa_path);

add_block('simulink/Sources/From Workspace', sa_path, ...
    'Position',              sa_pos,                 ...
    'VariableName',          'sa_ts',                ...
    'Interpolate',           'off',                  ...
    'OutputAfterFinalValue', 'Holding final value',  ...
    'OutDataTypeStr',        'double',               ...
    'SampleTime',            '0');

if ~isempty(goto1_path)
    sa_ph    = get_param(sa_path,    'PortHandles');
    goto1_ph = get_param(goto1_path, 'PortHandles');
    add_line(hifi_mdl, sa_ph.Outport(1), goto1_ph.Inport(1), 'autorouting','on');
    fprintf('  Spark Advance Constant → From Workspace (sa_ts), wired to Goto1\n');
else
    fprintf('  WARNING: Goto1 not found – SA From Workspace left unconnected\n');
end

% ---- A3. Signal logging --------------------------------------------------
enable_log(hifi_mdl, [hifi_mdl '/Vehicle Dynamics'], 1, 'Speed');
enable_log(hifi_mdl, [hifi_mdl '/Combustion'],        1, 'Torque');
set_param(hifi_mdl, 'SignalLogging',     'on');
set_param(hifi_mdl, 'SignalLoggingName', 'logsout');
set_param(hifi_mdl, 'StopTime',          '25');
fprintf('  Signal logging: Speed (VD out1), Torque (Combustion out1)\n');

save_system(hifi_mdl, hifi_file);
fprintf('[OK] Saved: %s\n', hifi_file);

% =========================================================================
%% PART B – Build enginespeed_qat_sfun.slx
% =========================================================================
fprintf('\n=== Part B: Building %s ===\n', sfun_mdl);

if bdIsLoaded(sfun_mdl), close_system(sfun_mdl, 0); end
copyfile(hifi_file, sfun_file, 'f');
load_system(sfun_file);

% ---- B1. Locate key blocks dynamically -----------------------------------
% Use name-matching (not hardcoded paths) for robustness across model versions.
comb_path  = [sfun_mdl '/Combustion'];

assert(~isempty(find_system(sfun_mdl,'SearchDepth',1,'Name','Combustion')), ...
       'Combustion block not found in %s', sfun_mdl);

% Find Vehicle Dynamics and Spark Advance by partial name match
all_blks = find_system(sfun_mdl, 'SearchDepth', 1, 'Type', 'block');
vd_path    = '';
sa_fw_path = '';
for k = 1:numel(all_blks)
    nm = lower(get_param(all_blks{k}, 'Name'));
    if contains(nm, 'vehicle') && contains(nm, 'dynamic')
        vd_path = all_blks{k};
    elseif contains(nm, 'spark') && contains(nm, 'advance')
        sa_fw_path = all_blks{k};
    end
end
assert(~isempty(vd_path),    'Vehicle Dynamics block not found in %s', sfun_mdl);
assert(~isempty(sa_fw_path), 'Spark Advance block not found in %s',    sfun_mdl);
fprintf('  VD block: %s\n',    vd_path);
fprintf('  SA block: %s\n',    sa_fw_path);

comb_pos = get_param(comb_path, 'Position');

% BEFORE deleting Combustion, capture the AirCharge source port handle
% (= output port of whatever block feeds Combustion's inport 1).
comb_ph_pre = get_param(comb_path, 'PortHandles');
ac_line     = get_param(comb_ph_pre.Inport(1), 'Line');
if ac_line > 0
    ac_src_ph = get_param(ac_line, 'SrcPortHandle');
    ac_src_blk = get_param(ac_src_ph, 'Parent');
    fprintf('  AirCharge source: %s\n', ac_src_blk);
else
    error('No line found on Combustion inport 1 – cannot trace AirCharge source.');
end
% comb_pos = [left, top, right, bottom]

% ---- B2. Delete Combustion -----------------------------------------------
comb_ph = get_param(comb_path, 'PortHandles');
for p = 1:numel(comb_ph.Inport)
    delete_line_at_port(comb_ph.Inport(p));
end
for p = 1:numel(comb_ph.Outport)
    delete_lines_at_outport(comb_ph.Outport(p));
end
delete_block(comb_path);
fprintf('  Deleted Combustion subsystem\n');

% ---- B3. Delete orphaned Goto1 -------------------------------------------
% SA From Workspace now feeds Goto1 but Combustion (which had From Spark
% Advance inside) is gone.  We'll wire SA directly to the Mux, so Goto1
% is no longer needed.
goto1_sf = find_block_type(sfun_mdl, 'Goto', 1);
if ~isempty(goto1_sf)
    goto1_sf_ph = get_param(goto1_sf, 'PortHandles');
    delete_line_at_port(goto1_sf_ph.Inport(1));   % SA FW → Goto1
    delete_block(goto1_sf);
    fprintf('  Deleted orphaned Goto1\n');
end

% ---- B4. Add Mux (3→1 vector) --------------------------------------------
% Place Mux to the left of where Combustion was.
mux_w   = 15;
mux_h   = comb_pos(4) - comb_pos(2);          % same height as old Combustion
mux_l   = comb_pos(1) - 100;
mux_pos = [mux_l, comb_pos(2), mux_l + mux_w, comb_pos(4)];

mux_blk = [sfun_mdl '/ROM_Mux'];
add_block('simulink/Signal Routing/Mux', mux_blk, ...
    'Inputs',   '3',    ...
    'Position', mux_pos);
fprintf('  Added Mux (3 inputs)\n');

% ---- B5. Add S-Function block --------------------------------------------
% Place at Combustion's old position.
sfun_blk = [sfun_mdl '/QAT ROM'];
add_block('simulink/User-Defined Functions/S-Function', sfun_blk, ...
    'FunctionName',  'sfun_rom_lstm_qat', ...
    'Position',       comb_pos,           ...
    'BackgroundColor','cyan');
fprintf('  Added S-Function block: sfun_rom_lstm_qat\n');

% Verify ports resolved (requires mexmaca64 on path)
sfun_ph = get_param(sfun_blk, 'PortHandles');
n_in    = numel(sfun_ph.Inport);
n_out   = numel(sfun_ph.Outport);
fprintf('  S-Function ports: in=%d, out=%d\n', n_in, n_out);
if n_in < 1 || n_out < 1
    error(['S-Function ports not resolved. Ensure sfun_rom_lstm_qat.mexmaca64 ' ...
           'is on the MATLAB path.\n  SRC = %s\n'], SRC);
end

% ---- B6. Wire the network ------------------------------------------------
%
%   AirCharge source (= Induction Delay out, captured before Combustion deleted)
%                         ─────────────────> Mux in[1]
%   SA From Workspace ───────────────────> Mux in[2]
%   VehicleDynamics Speed out ───────────> Mux in[3]
%
%   Mux out ──────────────────────────> S-Function in[1]  (width-3 vector)
%   S-Function out (Torque) ──────────> VehicleDynamics in[1] (Teng)
%
% S-Function normalises u[0]=AirCharge, u[1]=Speed, u[2]=SparkAdv.
% Mux output is [in1; in2; in3], so:
%   Mux in[1] = AirCharge, Mux in[2] = Speed, Mux in[3] = SparkAdv

vd_ph    = get_param(vd_path,    'PortHandles');
sa_fw_ph = get_param(sa_fw_path, 'PortHandles');
mux_ph   = get_param(mux_blk,   'PortHandles');
sfun_ph  = get_param(sfun_blk,  'PortHandles');

% AirCharge (source traced before Combustion deleted) → Mux[1]
add_line(sfun_mdl, ac_src_ph, mux_ph.Inport(1), 'autorouting','on');
fprintf('  Wired: AirCharge source (%s) → Mux[1]\n', ac_src_blk);

% Speed (VD out) → Mux[2]
add_line(sfun_mdl, vd_ph.Outport(1), mux_ph.Inport(2), 'autorouting','on');
fprintf('  Wired: Vehicle Dynamics Speed → Mux[2]\n');

% SparkAdvance (SA FW out) → Mux[3]
add_line(sfun_mdl, sa_fw_ph.Outport(1), mux_ph.Inport(3), 'autorouting','on');
fprintf('  Wired: SA From Workspace → Mux[3] (SparkAdv)\n');

% Mux → S-Function in[1]
add_line(sfun_mdl, mux_ph.Outport(1), sfun_ph.Inport(1), 'autorouting','on');
fprintf('  Wired: Mux → S-Function in[1] (width-3 vector)\n');

% S-Function Torque → VD in[1]  (re-fetch vd_ph in case autorouting moved it)
vd_ph2 = get_param(vd_path, 'PortHandles');
add_line(sfun_mdl, sfun_ph.Outport(1), vd_ph2.Inport(1), 'autorouting','on');
fprintf('  Wired: S-Function Torque → Vehicle Dynamics in[1] (Teng)\n');

% ---- B7. Signal logging -------------------------------------------------
enable_log(sfun_mdl, vd_path,   1, 'Speed');
enable_log(sfun_mdl, sfun_blk,  1, 'Torque');
set_param(sfun_mdl, 'SignalLogging',     'on');
set_param(sfun_mdl, 'SignalLoggingName', 'logsout');
set_param(sfun_mdl, 'StopTime',          '25');
fprintf('  Signal logging: Speed (VD out1), Torque (S-Function out1)\n');

save_system(sfun_mdl, sfun_file);
fprintf('[OK] Saved: %s\n', sfun_file);

if bdIsLoaded(hifi_mdl),  close_system(hifi_mdl,  0); end
if bdIsLoaded(sfun_mdl),  close_system(sfun_mdl,  0); end
warning('on','all');

fprintf('\n=== Both models built successfully ===\n');
fprintf('  HiFi  : %s\n', hifi_file);
fprintf('  ROM   : %s\n', sfun_file);
fprintf('\nNext: run scripts/run_closed_loop_validation.m\n');


%% ==========================================================================
%  Helper functions
%% ==========================================================================

function delete_line_at_port(port_handle)
%DELETE_LINE_AT_PORT  Delete line connected to an inport. Safe no-op if none.
    try
        ln = get_param(port_handle, 'Line');
        if ln > 0, delete_line(ln); end
    catch; end
end

function delete_lines_at_outport(port_handle)
%DELETE_LINES_AT_OUTPORT  Delete line(s) from an outport.
    try
        ln = get_param(port_handle, 'Line');
        if ln > 0, delete_line(ln); end
    catch; end
end

function blk = find_block_type(mdl, block_type, depth)
%FIND_BLOCK_TYPE  First block of given BlockType at given SearchDepth.
    candidates = find_system(mdl, 'SearchDepth', depth, 'BlockType', block_type);
    if isempty(candidates), blk = ''; else, blk = candidates{1}; end
end

function enable_log(mdl, block_path, port_idx, sig_name) %#ok<INUSL>
%ENABLE_LOG  Enable signal logging on a block output port (R2025b pattern).
    try
        ph   = get_param(block_path, 'PortHandles');
        pout = ph.Outport(port_idx);
        set_param(pout, 'DataLogging',         'on');
        set_param(pout, 'DataLoggingNameMode', 'Custom');
        set_param(pout, 'DataLoggingName',     sig_name);
    catch ME
        warning('enable_log: %s – %s', block_path, ME.message);
    end
end
