%% collect_validation_data.m
% Collects validation data with UNSEEN spark advance values and throttle profiles.
% SA values: 7, 17, 27 degrees (not in training set: 5,10,15,20,25,30)
% Throttle: ramp-based and sinusoidal-like profiles (different from training steps)

clear; close all; clc;

%% Configuration
proj_dir = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
data_dir = fullfile(proj_dir, 'data');

orig_model = fullfile(proj_dir, 'enginespeed.slx');
dc_model   = fullfile(proj_dir, 'enginespeed_val.slx');
dc_name    = 'enginespeed_val';

dt    = 0.05;
t_end = 25.0;
t_vec = (0:dt:t_end)';

% Unseen spark advance values
spark_vals_val = [7, 17, 27];

fprintf('=== Engine ROM Validation Data Collection ===\n');
fprintf('Using UNSEEN SA values: %s\n', num2str(spark_vals_val));
fprintf('Simulations: %d SA x 2 profiles\n\n', length(spark_vals_val));

%% Build validation throttle profiles (different character from training)
[p1, p2] = build_val_profiles(t_vec);
profiles = {p1, p2};
profile_names = {'Ramp_Steps', 'Wide_Sweep'};

%% Setup model copy
if exist(dc_model, 'file'), delete(dc_model); end
load_system(orig_model);
save_system('enginespeed', dc_model);
close_system('enginespeed', 0);
load_system(dc_model);
setup_model_for_datacollection(dc_name, dt);

%% Run simulations
all_records = [];
sim_idx = 0;

for i_sa = 1:length(spark_vals_val)
    sa = spark_vals_val(i_sa);
    set_param([dc_name '/Spark Advance'], 'Value', num2str(sa));

    for i_prof = 1:length(profiles)
        sim_idx = sim_idx + 1;
        throttle_sig = profiles{i_prof};

        fprintf('  [%d/%d] SA=%2d deg | Profile: %s ... ', ...
            sim_idx, length(spark_vals_val)*2, sa, profile_names{i_prof});

        throttle_ts = [t_vec, throttle_sig];
        assignin('base', 'throttle_ts', throttle_ts);

        try
            simOut = sim(dc_name, 'StopTime', num2str(t_end));
            [ac, spd, tq] = extract_signals(simOut, t_vec);

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

%% Save
fprintf('\nSaving validation data...\n');
save_csv(fullfile(data_dir, 'validation_data.csv'), ...
    'Time,AirCharge,Speed,Torque,SparkAdvance,SimID,Throttle', all_records);
save(fullfile(data_dir, 'validation_data.mat'), 'all_records', 't_vec', 'spark_vals_val');

fprintf('Total samples: %d\n', size(all_records,1));
fprintf('Saved to: %s\n', data_dir);

close_system(dc_name, 0);
if exist(dc_model, 'file'), delete(dc_model); end
fprintf('Validation data collection complete.\n');


%% ======================= HELPER FUNCTIONS =======================

function setup_model_for_datacollection(model_name, dt)
    try
        pos = get_param([model_name '/Throttle (degrees)'], 'Position');
    catch
        pos = [30, 135, 130, 165];
    end
    try
        delete_line(model_name, 'Throttle (degrees)/1', 'Throttle & Manifold/1');
    catch; end
    try
        delete_block([model_name '/Throttle (degrees)']);
    catch; end

    % R2025b: use 'Interpolate','off' instead of 'ZeroOrderHold','on'
    add_block('simulink/Sources/From Workspace', ...
        [model_name '/Throttle_FW'], ...
        'Position', pos, ...
        'VariableName', 'throttle_ts', ...
        'SampleTime',   '0', ...
        'Interpolate',  'off', ...
        'OutputAfterFinalValue', 'Holding final value', ...
        'OutDataTypeStr','double');
    add_line(model_name, 'Throttle_FW/1', 'Throttle & Manifold/1', ...
        'autorouting', 'on');

    set_param(model_name, 'SignalLogging',     'on');
    set_param(model_name, 'SignalLoggingName', 'logsout');
    enable_log(model_name, 'Throttle & Manifold', 1, 'AirCharge');
    enable_log(model_name, 'Combustion',          1, 'Torque');
    enable_log(model_name, 'Vehicle Dynamics',    1, 'Speed');
    save_system(model_name);
end

function enable_log(model, block, port_idx, sig_name)
    % R2025b: DataLogging properties live on the port handle, not line handle
    try
        ph   = get_param([model '/' block], 'PortHandles');
        pout = ph.Outport(port_idx);
        set_param(pout, 'DataLogging',         'on');
        set_param(pout, 'DataLoggingNameMode', 'Custom');
        set_param(pout, 'DataLoggingName',     sig_name);
    catch err
        warning('Signal logging for %s: %s', sig_name, err.message);
    end
end

function [ac, spd, tq] = extract_signals(simOut, t_ref)
    logsout = simOut.logsout;
    ac_el  = logsout.getElement('AirCharge');
    spd_el = logsout.getElement('Speed');
    tq_el  = logsout.getElement('Torque');

    t_raw   = ac_el.Values.Time;
    ac_raw  = squeeze(ac_el.Values.Data);
    spd_raw = squeeze(spd_el.Values.Data);
    tq_raw  = squeeze(tq_el.Values.Data);

    ac  = interp1(t_raw, ac_raw,  t_ref, 'linear', 'extrap');
    spd = interp1(t_raw, spd_raw, t_ref, 'linear', 'extrap');
    tq  = interp1(t_raw, tq_raw,  t_ref, 'linear', 'extrap');
end

function [p1, p2] = build_val_profiles(t_vec)
    t_end = t_vec(end);

    % Profile 1: Ramp-hold patterns (different character from training steps)
    lvl1 = [5, 15, 30, 20, 45, 10, 38, 25, 50, 12, 40, 18, 35, 8, 22];
    tsw1 = [0, 1.0, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5, 18.0, 20.0, 21.5, 23.0, 24.5];
    p1 = step_sig(t_vec, tsw1, lvl1);

    % Profile 2: Wide sweep with different timing
    lvl2 = [30, 8, 50, 5, 60, 15, 40, 10, 55, 20, 35, 12, 48, 25, 10];
    tsw2 = [0, 1.5, 3.5, 5.5, 7.0, 9.5, 11.5, 13.5, 15.5, 17.0, 19.0, 21.0, 22.5, 23.5, 24.5];
    p2 = step_sig(t_vec, tsw2, lvl2);
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
