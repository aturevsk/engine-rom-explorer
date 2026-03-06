%% collect_training_data.m
% Collects training data from enginespeed Simulink model for ROM development.
% Strategy: 6 Spark Advance values × 2 throttle profiles = 12 simulations
% Each simulation: 25 seconds with rich multi-step throttle excitation
% Outputs: data/training_data.csv and data/training_data.mat

clear; close all; clc;

%% Configuration
proj_dir = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
data_dir = fullfile(proj_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

orig_model = fullfile(proj_dir, 'enginespeed.slx');
dc_model   = fullfile(proj_dir, 'enginespeed_dc.slx');
dc_name    = 'enginespeed_dc';

dt    = 0.05;   % sample time [s]
t_end = 25.0;   % simulation duration per run [s]
t_vec = (0:dt:t_end)';

% Spark advance sweep [degrees] - training values
spark_vals = [5, 10, 15, 20, 25, 30];

fprintf('=== Engine ROM Training Data Collection ===\n');
fprintf('Simulations: %d SA values x 2 profiles = %d total\n', ...
    length(spark_vals), length(spark_vals)*2);
fprintf('Duration per sim: %.1f s | Sample dt: %.3f s\n\n', t_end, dt);

%% Build throttle excitation profiles
[p1, p2] = build_profiles(t_vec);
profiles = {p1, p2};
profile_names = {'MultiStep_A', 'MultiStep_B'};

%% Set up modified model
fprintf('Creating data-collection model copy...\n');
if exist(dc_model, 'file'), delete(dc_model); end

% Copy the model
load_system(orig_model);
save_system('enginespeed', dc_model);
close_system('enginespeed', 0);

% Load and modify the copy
load_system(dc_model);
setup_model_for_datacollection(dc_name, dt);
fprintf('Model ready.\n\n');

%% Run simulations
all_records = [];
sim_idx = 0;

for i_sa = 1:length(spark_vals)
    sa = spark_vals(i_sa);
    set_param([dc_name '/Spark Advance'], 'Value', num2str(sa));

    for i_prof = 1:length(profiles)
        sim_idx = sim_idx + 1;
        throttle_sig = profiles{i_prof};

        fprintf('  [%2d/%2d] SA = %2d deg | Profile: %s ... ', ...
            sim_idx, length(spark_vals)*length(profiles), ...
            sa, profile_names{i_prof});

        % Place throttle data in base workspace (From Workspace format)
        throttle_ts = [t_vec, throttle_sig];
        assignin('base', 'throttle_ts', throttle_ts);

        try
            simOut = sim(dc_name, 'StopTime', num2str(t_end));

            % Extract signals
            [ac, spd, tq] = extract_signals(simOut, t_vec);

            % Assemble record: [t, AirCharge, Speed, Torque, SA, SimID, Throttle]
            N = length(t_vec);
            block = [t_vec, ac, spd, tq, ...
                     sa*ones(N,1), sim_idx*ones(N,1), throttle_sig];
            all_records = [all_records; block]; %#ok<AGROW>
            fprintf('OK (%d samples)\n', N);

        catch err
            fprintf('FAILED: %s\n', err.message);
        end
    end
end

%% Save data
fprintf('\nSaving training data...\n');
save_csv(fullfile(data_dir, 'training_data.csv'), ...
    'Time,AirCharge,Speed,Torque,SparkAdvance,SimID,Throttle', all_records);
save(fullfile(data_dir, 'training_data.mat'), 'all_records', 't_vec', 'spark_vals', 'dt');

fprintf('Total samples: %d  |  Total simulations: %d\n', ...
    size(all_records,1), sim_idx);
fprintf('Saved to: %s\n', data_dir);

%% Cleanup
close_system(dc_name, 0);
if exist(dc_model, 'file'), delete(dc_model); end
fprintf('\nData collection complete.\n');


%% ======================= HELPER FUNCTIONS =======================

function setup_model_for_datacollection(model_name, dt)
    %% Replace Throttle (degrees) constant with From Workspace
    try
        pos = get_param([model_name '/Throttle (degrees)'], 'Position');
    catch
        pos = [30, 135, 130, 165];
    end

    % Remove existing connection from Throttle constant
    try
        delete_line(model_name, 'Throttle (degrees)/1', 'Throttle & Manifold/1');
    catch; end

    % Delete the constant block
    try
        delete_block([model_name '/Throttle (degrees)']);
    catch; end

    % Add From Workspace block in its place
    % Note: In R2025b, ZeroOrderHold and OutputDataType params were removed.
    % Use 'Interpolate','off' + 'OutputAfterFinalValue','HoldLastValue' for
    % zero-order hold behavior (Extrapolation requires Interpolate=on).
    add_block('simulink/Sources/From Workspace', ...
        [model_name '/Throttle_FW'], ...
        'Position', pos, ...
        'VariableName', 'throttle_ts', ...
        'SampleTime',   '0', ...
        'Interpolate',  'off', ...
        'OutputAfterFinalValue', 'Holding final value', ...
        'OutDataTypeStr','double');

    % Connect From Workspace → Throttle & Manifold input 1
    add_line(model_name, 'Throttle_FW/1', 'Throttle & Manifold/1', ...
        'autorouting', 'on');

    %% Enable signal logging via port handles
    set_param(model_name, 'SignalLogging',     'on');
    set_param(model_name, 'SignalLoggingName', 'logsout');

    enable_log(model_name, 'Throttle & Manifold', 1, 'AirCharge');
    enable_log(model_name, 'Combustion',          1, 'Torque');
    enable_log(model_name, 'Vehicle Dynamics',    1, 'Speed');

    save_system(model_name);
end

function enable_log(model, block, port_idx, sig_name)
    % In R2025b, DataLogging parameters live on the port handle, not the line handle.
    try
        ph = get_param([model '/' block], 'PortHandles');
        pout = ph.Outport(port_idx);
        set_param(pout, 'DataLogging',         'on');
        set_param(pout, 'DataLoggingNameMode', 'Custom');
        set_param(pout, 'DataLoggingName',     sig_name);
    catch err
        warning('Signal logging setup for %s failed: %s', sig_name, err.message);
    end
end

function [ac, spd, tq] = extract_signals(simOut, t_ref)
    logsout = simOut.logsout;

    ac_el  = logsout.getElement('AirCharge');
    spd_el = logsout.getElement('Speed');
    tq_el  = logsout.getElement('Torque');

    t_raw  = ac_el.Values.Time;
    ac_raw  = squeeze(ac_el.Values.Data);
    spd_raw = squeeze(spd_el.Values.Data);
    tq_raw  = squeeze(tq_el.Values.Data);

    % Resample to uniform grid
    ac  = interp1(t_raw, ac_raw,  t_ref, 'linear', 'extrap');
    spd = interp1(t_raw, spd_raw, t_ref, 'linear', 'extrap');
    tq  = interp1(t_raw, tq_raw,  t_ref, 'linear', 'extrap');
end

function [p1, p2] = build_profiles(t_vec)
    t_end = t_vec(end);

    % Profile 1: Rich multi-level steps covering low to high throttle
    lvl1  = [8, 25, 12, 40, 15, 30, 20, 50, 8, 35, 18, 45, 10, 28, 15];
    tsw1  = linspace(0, t_end, length(lvl1)+1);
    p1 = step_sig(t_vec, tsw1(1:end-1), lvl1);

    % Profile 2: Different ordering and range
    lvl2  = [20, 5, 35, 10, 55, 8, 28, 15, 42, 12, 32, 22, 18, 45, 6];
    tsw2  = linspace(0, t_end, length(lvl2)+1);
    p2 = step_sig(t_vec, tsw2(1:end-1), lvl2);
end

function sig = step_sig(t_vec, switch_times, levels)
    sig = zeros(size(t_vec));
    for i = 1:length(switch_times)
        if i < length(switch_times)
            idx = t_vec >= switch_times(i) & t_vec < switch_times(i+1);
        else
            idx = t_vec >= switch_times(i);
        end
        sig(idx) = levels(i);
    end
end

function save_csv(filepath, header, data)
    fid = fopen(filepath, 'w');
    fprintf(fid, '%s\n', header);
    for i = 1:size(data, 1)
        fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.4f,%d,%.4f\n', data(i,:));
    end
    fclose(fid);
end
