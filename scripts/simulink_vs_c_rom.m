%% simulink_vs_c_rom.m
% =========================================================================
% Fresh end-to-end validation: run enginespeed.slx NOW, feed the SAME
% inputs to each Pareto-optimal C ROM binary, compare outputs.
%
% Workflow:
%   1. Run enginespeed.slx for SA = 7°, 17°, 27° with a test throttle profile
%   2. Write Simulink inputs & ground-truth Torque to a temp CSV
%   3. Call the pre-compiled C binary (validate_roms) on that CSV
%   4. Parse C ROM per-step predictions from binary stdout
%   5. Plot Simulink vs C ROM torque traces + error bands
%   6. Save metrics to data/simulink_c_rom_results.json
%   7. Save publication-quality plots to plots/
%
% Requires:
%   - validate_roms binary (compiled by scripts/validate_c_code.py)
%   - enginespeed.slx in PROJ directory
% =========================================================================

clear; close all; clc;
warning('off','all');

%% ── Paths ────────────────────────────────────────────────────────────────
PROJ     = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
BIN      = fullfile(PROJ, 'validate_roms');        % compiled C binary
ORIG_MDL = fullfile(PROJ, 'enginespeed.slx');
DC_MDL   = fullfile(PROJ, 'enginespeed_cval.slx'); % temp copy
DC_NAME  = 'enginespeed_cval';
PLOT_DIR = fullfile(PROJ, 'plots');
DATA_DIR = fullfile(PROJ, 'data');

if ~exist(BIN, 'file')
    error('C binary not found: %s\n  Run: python3 scripts/validate_c_code.py first.', BIN);
end
if ~exist(ORIG_MDL, 'file')
    error('Simulink model not found: %s', ORIG_MDL);
end

%% ── Simulation config ────────────────────────────────────────────────────
dt    = 0.05;          % 50 ms timestep (matches training)
t_end = 25.0;          % 25 s simulation
t_vec = (0:dt:t_end)'; % 501 samples per run
N     = length(t_vec);

% Spark advance scenarios to test (3 from validation set)
spark_angles = [7, 17, 27];

% Single representative throttle profile (broad-excitation)
lvl = [5, 15, 30, 20, 45, 10, 38, 25, 50, 12, 40, 18, 35, 8, 22];
tsw = [0, 1.0, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5, 18.0, 20.0, 21.5, 23.0, 24.5];
throttle_profile = step_sig(t_vec, tsw, lvl);

%% ── Model ROM labels ─────────────────────────────────────────────────────
MODEL_KEYS   = {'narx_ridge', 'lstm_8', 'delta', 'lstm_16_q16', 'qat_lstm32'};
MODEL_LABELS = {'NARX-Ridge (0.38 KB)', 'LSTM-8 (2.52 KB)', ...
                'Delta Composite (2.79 KB)', 'LSTM-16 Q16 (4.05 KB)', ...
                'QAT LSTM-32 (6.97 KB)'};
MODEL_COLORS = {[0.5 0.5 0.5], [0.122 0.467 0.706], [0.890 0.467 0.761], ...
                [0.090 0.745 0.812], [0.549 0.337 0.294]};

%% ── Setup Simulink model copy ────────────────────────────────────────────
fprintf('=== Simulink vs C ROM Validation ===\n');
fprintf('Setting up Simulink model...\n');

if exist(DC_MDL, 'file'), delete(DC_MDL); end
load_system(ORIG_MDL);
save_system('enginespeed', DC_MDL);
close_system('enginespeed', 0);
load_system(DC_MDL);
setup_model(DC_NAME, dt);
fprintf('Model ready: %s\n\n', DC_NAME);

%% ── Run Simulink + C ROM for each SA scenario ────────────────────────────
all_results = struct();   % collected per scenario

for i_sa = 1:length(spark_angles)
    sa = spark_angles(i_sa);
    fprintf('--- SA = %d deg ---\n', sa);

    %% 1. Run Simulink
    set_param([DC_NAME '/Spark Advance'], 'Value', num2str(sa));
    throttle_ts = [t_vec, throttle_profile];
    assignin('base', 'throttle_ts', throttle_ts);

    fprintf('  Running Simulink...');
    simOut = sim(DC_NAME, 'StopTime', num2str(t_end));
    [ac_sim, spd_sim, tq_sim] = extract_signals(simOut, t_vec);
    fprintf(' done (%d samples, Torque range [%.1f, %.1f] N·m)\n', ...
            N, min(tq_sim), max(tq_sim));

    %% 2. Write inputs + Simulink ground truth to temp CSV
    tmp_csv = fullfile(DATA_DIR, sprintf('tmp_cval_sa%d.csv', sa));
    sim_id  = i_sa;
    write_csv(tmp_csv, t_vec, ac_sim, spd_sim, tq_sim, sa, sim_id, throttle_profile);
    fprintf('  Wrote %d rows to %s\n', N, tmp_csv);

    %% 3. Call C binary
    tmp_out = fullfile(DATA_DIR, sprintf('tmp_cval_sa%d_out.txt', sa));
    cmd = sprintf('"%s" "%s" > "%s" 2>&1', BIN, tmp_csv, tmp_out);
    fprintf('  Running C binary...');
    status = system(cmd);
    if status ~= 0
        fprintf(' FAILED (exit %d)\n', status);
        continue;
    end
    fprintf(' done\n');

    %% 4. Parse C binary output
    [data_mat, metrics_c] = parse_c_output(tmp_out, sim_id);
    if isempty(data_mat)
        fprintf('  WARNING: no DATA lines parsed\n');
        continue;
    end

    % data_mat columns: sim_id, t, true_tq, narx, lstm8, delta, q16, qat
    t_c   = data_mat(:,2);
    tq_c  = data_mat(:,3);  % = Simulink ground truth (sanity check)
    narx  = data_mat(:,4);
    l8    = data_mat(:,5);
    delt  = data_mat(:,6);
    q16   = data_mat(:,7);
    qat   = data_mat(:,8);

    pred_mat = [narx, l8, delt, q16, qat];

    fprintf('  C ROM results (vs Simulink Torque):\n');
    for k = 1:length(MODEL_KEYS)
        err   = pred_mat(:,k) - tq_sim;
        r_c   = sqrt(mean(err.^2));
        r2_c  = 1 - sum(err.^2) / sum((tq_sim - mean(tq_sim)).^2);
        fprintf('    %-20s  RMSE = %.4f N·m  R² = %.6f\n', ...
                MODEL_LABELS{k}, r_c, r2_c);
        metrics_c.(MODEL_KEYS{k}).rmse = r_c;
        metrics_c.(MODEL_KEYS{k}).r2   = r2_c;
    end

    %% 5. Store for plotting
    all_results.(sprintf('sa%d', sa)) = struct( ...
        'sa', sa, 't', t_vec, 'tq_sim', tq_sim, ...
        'ac', ac_sim, 'spd', spd_sim, 'throttle', throttle_profile, ...
        'pred', pred_mat, 'metrics', metrics_c);

    %% Clean up temp files
    if exist(tmp_csv, 'file'), delete(tmp_csv); end
    if exist(tmp_out, 'file'), delete(tmp_out); end
end

close_system(DC_NAME, 0);
if exist(DC_MDL, 'file'), delete(DC_MDL); end

%% ── PLOTS ────────────────────────────────────────────────────────────────
sa_fields = fieldnames(all_results);
n_sa = length(sa_fields);
if n_sa == 0
    error('No simulation results collected. Check Simulink model setup.');
end

fprintf('\nGenerating plots...\n');

%% Plot A: Time traces — 3×2 grid (one row per SA, col1=torque col2=error)
fig1 = figure('Position', [50 50 1400 320*n_sa], 'Visible', 'off');

for i = 1:n_sa
    fld = sa_fields{i};
    res = all_results.(fld);
    t   = res.t;
    tq  = res.tq_sim;
    sa  = res.sa;

    % Column 1: Torque traces
    ax1 = subplot(n_sa, 2, 2*i-1);
    p_sim = plot(t, tq, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Simulink (ground truth)');
    hold on;
    for k = 1:length(MODEL_KEYS)
        plot(t, res.pred(:,k), '-', 'LineWidth', 1.2, ...
             'Color', MODEL_COLORS{k}, 'DisplayName', MODEL_LABELS{k});
    end
    hold off;
    grid on;
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Torque (N·m)', 'FontSize', 10);
    title(sprintf('SA = %d° — Torque: Simulink vs C ROM', sa), ...
          'FontSize', 11, 'FontWeight', 'bold');

    rmse_strs = cell(1, length(MODEL_KEYS));
    for k = 1:length(MODEL_KEYS)
        rmse_strs{k} = sprintf('%s: %.3f', strtok(MODEL_LABELS{k}), res.metrics.(MODEL_KEYS{k}).rmse);
    end
    text(0.02, 0.97, ['RMSE (N·m): ' strjoin(rmse_strs, '  ')], ...
         'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 7.5, 'BackgroundColor', [1 1 1 0.8]);
    if i == 1
        legend('Location', 'best', 'FontSize', 7);
    end

    % Column 2: Absolute error
    ax2 = subplot(n_sa, 2, 2*i);
    hold on;
    fill([t; flipud(t)], [0.91*ones(N,1); -0.91*ones(N,1)], ...
         [0.8 1.0 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.35, ...
         'DisplayName', '±0.91 N·m baseline band');
    for k = 1:length(MODEL_KEYS)
        err = res.pred(:,k) - tq;
        plot(t, err, '-', 'LineWidth', 1.0, 'Color', MODEL_COLORS{k}, ...
             'DisplayName', MODEL_LABELS{k});
    end
    plot(t, zeros(N,1), 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    hold off;
    grid on;
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('C ROM − Simulink (N·m)', 'FontSize', 10);
    title(sprintf('SA = %d° — Prediction Error', sa), ...
          'FontSize', 11, 'FontWeight', 'bold');
    if i == 1
        legend('Location', 'best', 'FontSize', 7);
    end
end

sgtitle('C ROM Validation Against Fresh Simulink Simulation — enginespeed.slx', ...
        'FontSize', 13, 'FontWeight', 'bold');

out_traces = fullfile(PLOT_DIR, 'simulink_c_rom_traces.png');
exportgraphics(fig1, out_traces, 'Resolution', 150);
fprintf('  Saved → plots/simulink_c_rom_traces.png\n');

%% Plot B: RMSE summary bar chart across all SA angles
all_rmse = zeros(length(MODEL_KEYS), n_sa);
all_r2   = zeros(length(MODEL_KEYS), n_sa);
sa_labels = cell(1, n_sa);

for i = 1:n_sa
    fld = sa_fields{i};
    sa_labels{i} = sprintf('SA=%d°', all_results.(fld).sa);
    for k = 1:length(MODEL_KEYS)
        all_rmse(k,i) = all_results.(fld).metrics.(MODEL_KEYS{k}).rmse;
        all_r2(k,i)   = all_results.(fld).metrics.(MODEL_KEYS{k}).r2;
    end
end

fig2 = figure('Position', [50 50 1200 480], 'Visible', 'off');

subplot(1,2,1);
b = bar(all_rmse', 'grouped');
for k = 1:length(MODEL_KEYS)
    b(k).FaceColor = MODEL_COLORS{k};
    b(k).EdgeColor = 'k';
    b(k).LineWidth = 0.7;
end
yline(0.91, '--', 'Float32 baseline 0.91 N·m', ...
      'Color', [0 0.55 0], 'LineWidth', 1.3, 'LabelVerticalAlignment', 'bottom');
set(gca, 'XTickLabel', sa_labels, 'FontSize', 10);
ylabel('RMSE (N·m)', 'FontSize', 11);
title({'C ROM Validation RMSE', '(vs Live Simulink Ground Truth)'}, ...
      'FontSize', 11, 'FontWeight', 'bold');
grid on; ylim([0 max(all_rmse(:))*1.25]);
legend(MODEL_LABELS, 'Location', 'northeast', 'FontSize', 7.5);
% Annotate bars
for i = 1:n_sa
    for k = 1:length(MODEL_KEYS)
        x = i + (k - (length(MODEL_KEYS)+1)/2) * 1/(length(MODEL_KEYS)+1);
        text(x, all_rmse(k,i)*1.04, sprintf('%.2f', all_rmse(k,i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 6.5, 'FontWeight', 'bold');
    end
end

subplot(1,2,2);
b2 = bar((1 - all_r2')*100, 'grouped');   % plot (1-R²)×100 so smaller = better
for k = 1:length(MODEL_KEYS)
    b2(k).FaceColor = MODEL_COLORS{k};
    b2(k).EdgeColor = 'k';
    b2(k).LineWidth = 0.7;
end
set(gca, 'XTickLabel', sa_labels, 'FontSize', 10);
ylabel('(1 - R²) × 100  [%]', 'FontSize', 11);
title({'Unexplained Variance', '(lower = better)'}, 'FontSize', 11, 'FontWeight', 'bold');
grid on;

sgtitle('C ROM vs Live Simulink — RMSE and R² by Spark Advance', ...
        'FontSize', 12, 'FontWeight', 'bold');

out_bar = fullfile(PLOT_DIR, 'simulink_c_rom_metrics.png');
exportgraphics(fig2, out_bar, 'Resolution', 150);
fprintf('  Saved → plots/simulink_c_rom_metrics.png\n');

%% Plot C: Scatter – C ROM vs Simulink across all scenarios
fig3 = figure('Position', [50 50 250*length(MODEL_KEYS) 280], 'Visible', 'off');
all_tq_sim  = [];
all_preds   = cell(1, length(MODEL_KEYS));
for k = 1:length(MODEL_KEYS)
    all_preds{k} = [];
end
for i = 1:n_sa
    fld = sa_fields{i};
    res = all_results.(fld);
    all_tq_sim = [all_tq_sim; res.tq_sim]; %#ok<AGROW>
    for k = 1:length(MODEL_KEYS)
        all_preds{k} = [all_preds{k}; res.pred(:,k)]; %#ok<AGROW>
    end
end
tq_rng = [min(all_tq_sim)-3, max(all_tq_sim)+3];

for k = 1:length(MODEL_KEYS)
    subplot(1, length(MODEL_KEYS), k);
    err_k = all_preds{k} - all_tq_sim;
    r_k   = sqrt(mean(err_k.^2));
    r2_k  = 1 - sum(err_k.^2)/sum((all_tq_sim-mean(all_tq_sim)).^2);

    scatter(all_tq_sim, all_preds{k}, 5, MODEL_COLORS{k}, 'filled', ...
            'MarkerFaceAlpha', 0.3);
    hold on;
    plot(tq_rng, tq_rng, 'k--', 'LineWidth', 1.0);
    hold off;
    xlabel('Simulink (N·m)', 'FontSize', 8);
    ylabel('C ROM (N·m)', 'FontSize', 8);
    title(MODEL_LABELS{k}, 'FontSize', 8, 'FontWeight', 'bold', 'Interpreter', 'none');
    xlim(tq_rng); ylim(tq_rng); axis square; grid on;
    text(0.05, 0.93, sprintf('RMSE=%.3f\nR²=%.5f', r_k, r2_k), ...
         'Units', 'normalized', 'FontSize', 7.5, ...
         'BackgroundColor', [1 1 1 0.85], 'VerticalAlignment', 'top');
end

sgtitle('C ROM vs Simulink Ground Truth — All SA Scenarios', ...
        'FontSize', 11, 'FontWeight', 'bold');

out_scatter = fullfile(PLOT_DIR, 'simulink_c_rom_scatter.png');
exportgraphics(fig3, out_scatter, 'Resolution', 150);
fprintf('  Saved → plots/simulink_c_rom_scatter.png\n');

%% ── Save JSON results ────────────────────────────────────────────────────
fprintf('\nSaving JSON results...\n');
json_out = struct();
json_out.description = ['Fresh Simulink vs C ROM validation. ' ...
    'Ground truth = enginespeed.slx simulation output. ' ...
    'C ROM predictions from compiled binary (validate_roms).'];
json_out.simulator   = 'enginespeed.slx (Simulink)';
json_out.dt_s        = dt;
json_out.t_end_s     = t_end;
json_out.n_steps_per_sim = N;

for i = 1:n_sa
    fld = sa_fields{i};
    res = all_results.(fld);
    s   = struct('sa_deg', res.sa, 'models', struct());
    for k = 1:length(MODEL_KEYS)
        mk = MODEL_KEYS{k};
        s.models.(mk) = struct( ...
            'rmse_Nm', res.metrics.(mk).rmse, ...
            'r2',      res.metrics.(mk).r2, ...
            'label',   MODEL_LABELS{k});
    end
    json_out.(fld) = s;
end

% Overall across all SA
json_out.overall = struct();
for k = 1:length(MODEL_KEYS)
    mk  = MODEL_KEYS{k};
    err = all_preds{k} - all_tq_sim;
    r_o = sqrt(mean(err.^2));
    r2o = 1 - sum(err.^2)/sum((all_tq_sim-mean(all_tq_sim)).^2);
    json_out.overall.(mk) = struct('rmse_Nm', r_o, 'r2', r2o, 'label', MODEL_LABELS{k});
    fprintf('  Overall %-20s  RMSE=%.4f N·m  R²=%.6f\n', MODEL_LABELS{k}, r_o, r2o);
end

json_out.plots = {
    'plots/simulink_c_rom_traces.png', ...
    'plots/simulink_c_rom_metrics.png', ...
    'plots/simulink_c_rom_scatter.png'};

json_str = jsonencode(json_out);
% Pretty-print
fid = fopen(fullfile(DATA_DIR, 'simulink_c_rom_results.json'), 'w');
fprintf(fid, '%s', json_str);
fclose(fid);
fprintf('  Saved → data/simulink_c_rom_results.json\n');

fprintf('\n=== Simulink vs C ROM Validation Complete ===\n');
fprintf('Plots saved to: %s\n', PLOT_DIR);


%% ============================= HELPERS ==================================

function setup_model(model_name, dt)
    try
        delete_line(model_name, 'Throttle (degrees)/1', 'Throttle & Manifold/1');
    catch; end
    try
        delete_block([model_name '/Throttle (degrees)']);
    catch; end
    try
        pos = [30, 135, 130, 165];
        add_block('simulink/Sources/From Workspace', ...
            [model_name '/Throttle_FW'], ...
            'Position',             pos, ...
            'VariableName',         'throttle_ts', ...
            'SampleTime',           '0', ...
            'Interpolate',          'off', ...
            'OutputAfterFinalValue','Holding final value', ...
            'OutDataTypeStr',       'double');
        add_line(model_name, 'Throttle_FW/1', 'Throttle & Manifold/1', ...
                 'autorouting', 'on');
    catch err
        warning('Throttle block setup: %s', err.message);
    end

    set_param(model_name, 'SignalLogging',     'on');
    set_param(model_name, 'SignalLoggingName', 'logsout');
    enable_log(model_name, 'Throttle & Manifold', 1, 'AirCharge');
    enable_log(model_name, 'Combustion',          1, 'Torque');
    enable_log(model_name, 'Vehicle Dynamics',    1, 'Speed');
    save_system(model_name);
end

function enable_log(model, block, port_idx, sig_name)
    try
        ph   = get_param([model '/' block], 'PortHandles');
        pout = ph.Outport(port_idx);
        set_param(pout, 'DataLogging',         'on');
        set_param(pout, 'DataLoggingNameMode', 'Custom');
        set_param(pout, 'DataLoggingName',     sig_name);
    catch err
        warning('Signal logging %s: %s', sig_name, err.message);
    end
end

function [ac, spd, tq] = extract_signals(simOut, t_ref)
    logsout = simOut.logsout;
    ac_el   = logsout.getElement('AirCharge');
    spd_el  = logsout.getElement('Speed');
    tq_el   = logsout.getElement('Torque');

    t_raw   = ac_el.Values.Time;
    ac_raw  = squeeze(ac_el.Values.Data);
    spd_raw = squeeze(spd_el.Values.Data);
    tq_raw  = squeeze(tq_el.Values.Data);

    ac  = interp1(t_raw, ac_raw,  t_ref, 'linear', 'extrap');
    spd = interp1(t_raw, spd_raw, t_ref, 'linear', 'extrap');
    tq  = interp1(t_raw, tq_raw,  t_ref, 'linear', 'extrap');
end

function write_csv(path, t, ac, spd, tq, sa, sim_id, throttle)
    fid = fopen(path, 'w');
    fprintf(fid, 'Time,AirCharge,Speed,Torque,SparkAdvance,SimID,Throttle\n');
    for i = 1:length(t)
        fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.4f,%d,%.4f\n', ...
                t(i), ac(i), spd(i), tq(i), sa, sim_id, throttle(i));
    end
    fclose(fid);
end

function [data_mat, metrics] = parse_c_output(filepath, sim_id)
    data_mat = [];
    metrics  = struct();
    fid = fopen(filepath, 'r');
    if fid < 0
        return;
    end
    while true
        line = fgetl(fid);
        if ~ischar(line), break; end
        line = strtrim(line);
        if startsWith(line, 'DATA ')
            nums = sscanf(line(6:end), '%f');
            if length(nums) >= 8
                data_mat = [data_mat; nums']; %#ok<AGROW>
            end
        elseif startsWith(line, 'OVERALL ')
            parts = strsplit(strtrim(line(9:end)));
            if length(parts) >= 4
                mk = strtrim(parts{1});
                mk = strrep(mk, '-', '_');
                metrics.(mk).n    = str2double(parts{2});
                metrics.(mk).rmse = str2double(parts{3});
                metrics.(mk).r2   = str2double(parts{4});
            end
        end
    end
    fclose(fid);
    % Filter to requested sim_id
    if ~isempty(data_mat) && size(data_mat,2) >= 1
        data_mat = data_mat(data_mat(:,1) == sim_id, :);
    end
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
