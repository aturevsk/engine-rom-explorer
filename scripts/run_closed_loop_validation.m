%% run_closed_loop_validation.m
% ==========================================================================
% Closed-Loop QAT ROM Validation – 5 Scenarios
%
% Follows the MathWorks ROM-validation example (reduced-order-modeling of
% subsystems in engine model): replace the Combustion subsystem with the
% QAT LSTM-32 ROM (C Caller block), run the full closed-loop simulation,
% and compare Engine Speed and Torque with the original HiFi model.
%
% Scenarios:
%   S1: Rich transient throttle, SA = 7°  (below-nominal spark)
%   S2: Rich transient throttle, SA = 15° (nominal)
%   S3: Rich transient throttle, SA = 27° (above-nominal spark)
%   S4: Throttle step (5→50°) at t=5s, SA = 15°  (load transient)
%   S5: SA step (7→27°) at t=5s, constant throttle = 30°
%
% Models required (built by create_sfun_validation_models.m):
%   enginespeed_hifi_val.slx   – HiFi ground truth
%   enginespeed_qat_sfun.slx   – ROM closed-loop (S-Function)
%
% Outputs:
%   plots/clval_S*.png         – per-scenario 3-panel comparison
%   plots/clval_summary.png    – summary bar chart + overlay
%   data/cl_validation_results.mat
% ==========================================================================

clc;

%% 0. Setup ----------------------------------------------------------------
PROJ      = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
SRC       = fullfile(PROJ, 'src');
PLOTS_DIR = fullfile(PROJ, 'plots');
DATA_DIR  = fullfile(PROJ, 'data');
cd(PROJ);

if ~any(strcmp(SRC, strsplit(path, pathsep))), addpath(SRC); end
if ~isfolder(PLOTS_DIR), mkdir(PLOTS_DIR); end
if ~isfolder(DATA_DIR),  mkdir(DATA_DIR);  end

hifi_mdl  = 'enginespeed_hifi_val';
cc_mdl    = 'enginespeed_qat_sfun';
hifi_file = fullfile(PROJ, [hifi_mdl '.slx']);
cc_file   = fullfile(PROJ, [cc_mdl   '.slx']);

% Build models if they don't exist yet
if ~isfile(hifi_file) || ~isfile(cc_file)
    fprintf('Models not found – building them now…\n');
    run(fullfile(PROJ, 'scripts', 'create_sfun_validation_models.m'));
    % Re-establish paths (create script does cd(PROJ))
    cd(PROJ);
end

assert(isfile(hifi_file), 'HiFi model not found: %s', hifi_file);
assert(isfile(cc_file),   'ROM model not found: %s',  cc_file);

% Suppress harmless Simulink warnings
warning('off', 'Simulink:blocks:DelaySmallerThanOneSampleTime');
warning('off', 'Simulink:blocks:NegativeDelayValue');
warning('off', 'Simulink:Engine:UndefinedOutput');
warning('off', 'Simulink:blocks:DelayExceedsMaxDelay');

%% 1. Define test scenarios ------------------------------------------------
T_sim = 25.0;                     % simulation duration [s]
dt    = 0.05;                     % time step matching LSTM training rate [s]
t_vec = (0:dt:T_sim)';            % 501 points on uniform grid

% Helper lambdas
const_sig = @(v) v * ones(size(t_vec));
step_sig  = @(levels, times) build_step_signal(t_vec, levels, times);

% Rich multi-step throttle profile (same as create_sfun_validation.m)
lvl1  = [5, 15, 30, 20, 45, 10, 38, 25, 50, 12, 40, 18, 35, 8, 22];
tsw1  = [0, 1.0, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, ...
         16.5, 18.0, 20.0, 21.5, 23.0, 24.5];
rich_thr = step_sig(lvl1, tsw1);

% Scenarios (struct array)
SC(1).name     = 'S1_Rich_SA7';
SC(1).label    = 'S1: Rich transient throttle,  SA = 7°';
SC(1).throttle = rich_thr;
SC(1).sa       = const_sig(7);

SC(2).name     = 'S2_Rich_SA15';
SC(2).label    = 'S2: Rich transient throttle,  SA = 15°';
SC(2).throttle = rich_thr;
SC(2).sa       = const_sig(15);

SC(3).name     = 'S3_Rich_SA27';
SC(3).label    = 'S3: Rich transient throttle,  SA = 27°';
SC(3).throttle = rich_thr;
SC(3).sa       = const_sig(27);

SC(4).name     = 'S4_Throttle_Step';
SC(4).label    = 'S4: Throttle step 5→50° at t=5s, then 50→5° at t=18s,  SA = 15°';
SC(4).throttle = step_sig([5, 50, 5], [0, 5, 18]);
SC(4).sa       = const_sig(15);

SC(5).name     = 'S5_SA_Step';
SC(5).label    = 'S5: SA step 7→27° at t=5s,  throttle = 30° constant';
SC(5).throttle = const_sig(30);
SC(5).sa       = step_sig([7, 27], [0, 5]);

n_sc = numel(SC);

%% 2. Load both models (keep open across all scenarios for speed) ----------
fprintf('Loading models…\n');
if ~bdIsLoaded(hifi_mdl), load_system(hifi_file); end
if ~bdIsLoaded(cc_mdl),   load_system(cc_file);   end

%% 3. Main scenario loop ---------------------------------------------------
R = struct();   % results

for i = 1:n_sc
    sc = SC(i);
    fprintf('\n--- Scenario %d/%d: %s ---\n', i, n_sc, sc.label);

    % Assign input signals to base workspace (both models read from there)
    throttle_ts = [t_vec, sc.throttle]; %#ok<NASGU>
    sa_ts       = [t_vec, sc.sa];       %#ok<NASGU>
    assignin('base', 'throttle_ts', throttle_ts);
    assignin('base', 'sa_ts',       sa_ts);

    % ---- 3a. HiFi simulation ----
    fprintf('  HiFi … ');
    try
        warning('off','all');
        simH = sim(hifi_mdl, 'StopTime', num2str(T_sim));
        warning('on','all');
        [t_h,    spd_h] = extract_signal(simH, 'Speed');
        [t_tq_h, tq_h]  = extract_signal(simH, 'Torque');
        fprintf('OK  (%d spd / %d tq samples)\n', numel(t_h), numel(t_tq_h));
    catch ME
        fprintf('FAILED: %s\n', ME.message);
        continue;
    end

    % ---- 3b. ROM simulation ----
    % NOTE: S-Function runs at discrete 0.05s → tq_r has 501 samples,
    %       while ode23 Speed has ~2500+ samples.  Each signal needs its
    %       own time vector; they are all interpolated onto t_vec (0.05s).
    fprintf('  ROM  … ');
    try
        warning('off','all');
        simR = sim(cc_mdl, 'StopTime', num2str(T_sim));
        warning('on','all');
        [t_r,    spd_r] = extract_signal(simR, 'Speed');
        [t_tq_r, tq_r]  = extract_signal(simR, 'Torque');
        fprintf('OK  (%d spd / %d tq samples)\n', numel(t_r), numel(t_tq_r));
    catch ME
        fprintf('FAILED: %s\n', ME.message);
        continue;
    end

    % ---- 3c. Time-align all signals onto uniform 0.05s grid --------------
    % HiFi:  ode23 variable-step for both speed and torque
    % ROM:   ode23 for speed, discrete 0.05s for torque (S-Function)
    % All → interpolated onto t_vec for apples-to-apples comparison.
    t_min = max([t_h(1), t_r(1), t_tq_h(1), t_tq_r(1)]);
    t_max = min([t_h(end), t_r(end), t_tq_h(end), t_tq_r(end)]);
    mask  = (t_vec >= t_min) & (t_vec <= t_max);
    t_cmp = t_vec(mask);

    spd_h_u  = interp1(t_h,    spd_h, t_cmp, 'linear', 'extrap');
    spd_r_u  = interp1(t_r,    spd_r, t_cmp, 'linear', 'extrap');
    tq_h_u   = interp1(t_tq_h, tq_h,  t_cmp, 'linear', 'extrap');
    tq_r_u   = interp1(t_tq_r, tq_r,  t_cmp, 'linear', 'extrap');

    % ---- 3d. Primary metric: Torque (N·m) --------------------------------
    % ROM directly predicts torque; this is the model accuracy test.
    err_tq    = tq_h_u - tq_r_u;
    rmse_tq   = sqrt(mean(err_tq.^2));
    ss_tot_tq = sum((tq_h_u - mean(tq_h_u)).^2);
    r2_tq     = 1 - sum(err_tq.^2) / max(ss_tot_tq, 1e-12);
    maxe_tq   = max(abs(err_tq));
    fprintf('  *** Torque RMSE = %.4f N·m  R² = %.5f  MaxErr = %.2f N·m ***\n', ...
        rmse_tq, r2_tq, maxe_tq);

    % ---- 3e. Secondary metric: Engine Speed (rad/s) ----------------------
    % Speed is a downstream result; validates closed-loop system response.
    err_spd    = spd_h_u - spd_r_u;
    rmse_spd   = sqrt(mean(err_spd.^2));
    ss_tot_spd = sum((spd_h_u - mean(spd_h_u)).^2);
    r2_spd     = 1 - sum(err_spd.^2) / max(ss_tot_spd, 1e-12);
    maxe_spd   = max(abs(err_spd));
    rmse_rpm   = rmse_spd * 60 / (2*pi);
    fprintf('      Speed RMSE  = %.4f rad/s (%.2f rpm)  R² = %.5f\n', ...
        rmse_spd, rmse_rpm, r2_spd);

    % ---- 3f. Store results ----
    R(i).name      = sc.name;
    R(i).label     = sc.label;
    R(i).t_h       = t_h;
    R(i).spd_h     = spd_h;
    R(i).t_tq_h    = t_tq_h;
    R(i).tq_h      = tq_h;
    R(i).t_r       = t_r;
    R(i).spd_r     = spd_r;
    R(i).t_tq_r    = t_tq_r;
    R(i).tq_r      = tq_r;
    R(i).t_cmp     = t_cmp;
    R(i).tq_h_u    = tq_h_u;
    R(i).tq_r_u    = tq_r_u;
    R(i).err_tq    = err_tq;
    R(i).rmse_tq   = rmse_tq;
    R(i).r2_tq     = r2_tq;
    R(i).maxe_tq   = maxe_tq;
    R(i).spd_h_u   = spd_h_u;
    R(i).spd_r_u   = spd_r_u;
    R(i).err_spd   = err_spd;
    R(i).rmse_spd  = rmse_spd;
    R(i).r2_spd    = r2_spd;
    R(i).rmse_rpm  = rmse_rpm;
    R(i).maxe_spd  = maxe_spd;
    R(i).throttle_prof = sc.throttle;
    R(i).sa_prof       = sc.sa;

    % ---- 3g. Per-scenario 4-panel plot -----------------------------------
    fig = figure('Visible','off','Position',[50 50 1100 860]);

    % Panel 1: Torque comparison (PRIMARY — direct ROM output)
    ax1 = subplot(4,1,1);
    plot(t_tq_h, tq_h, 'b-',  'LineWidth', 1.3, 'DisplayName', 'HiFi Combustion');
    hold on;
    plot(t_tq_r, tq_r, 'r--', 'LineWidth', 1.8, 'DisplayName', 'QAT ROM (S-Function)');
    ylabel('Torque (N·m)');
    title(sprintf('%s\nTorque RMSE = %.3f N·m | R^2 = %.5f', ...
        sc.label, rmse_tq, r2_tq), 'FontSize', 9);
    legend('Location','best','FontSize',8); grid on;
    xlim([0 T_sim]);

    % Panel 2: Torque error
    ax2 = subplot(4,1,2);
    plot(t_cmp, err_tq, 'k-', 'LineWidth', 0.8);
    hold on;
    yline( rmse_tq, 'r--', sprintf('+%.3f N·m', rmse_tq), ...
        'LabelHorizontalAlignment','right','FontSize',8);
    yline(-rmse_tq, 'r--', sprintf('-%.3f N·m', rmse_tq), ...
        'LabelHorizontalAlignment','right','FontSize',8);
    yline(0, 'k:', 'LineWidth',0.5);
    fill([t_cmp; flipud(t_cmp)], [err_tq; zeros(size(err_tq))], ...
         [0.9 0.7 0.7], 'FaceAlpha',0.3, 'EdgeColor','none');
    ylabel('Torque Error (N·m)');
    title('Torque Error  [HiFi − ROM]');
    grid on; xlim([0 T_sim]);

    % Panel 3: Engine Speed (closed-loop secondary check)
    ax3 = subplot(4,1,3);
    plot(t_h, spd_h, 'b-',  'LineWidth', 1.1, 'DisplayName', 'HiFi');
    hold on;
    plot(t_r, spd_r, 'r--', 'LineWidth', 1.5, 'DisplayName', 'ROM');
    ylabel('Engine Speed (rad/s)');
    title(sprintf('Closed-Loop Speed: RMSE = %.2f rpm | R^2 = %.5f', ...
        rmse_rpm, r2_spd), 'FontSize', 9);
    legend('Location','best','FontSize',8); grid on;
    xlim([0 T_sim]);

    % Panel 4: Inputs (throttle + SA dual y-axis)
    ax4 = subplot(4,1,4);
    yyaxis left;
    plot(t_vec, sc.throttle, 'm-', 'LineWidth',1.2); ylabel('Throttle (deg)');
    yyaxis right;
    plot(t_vec, sc.sa, 'c-', 'LineWidth',1.2); ylabel('Spark Advance (deg)');
    xlabel('Time (s)');
    title('ROM Inputs: Throttle & Spark Advance');
    grid on; xlim([0 T_sim]);

    linkaxes([ax1 ax2 ax3 ax4], 'x');
    sgtitle('QAT LSTM-32 ROM – Closed-Loop Validation', 'FontSize',12,'FontWeight','bold');

    out_png = fullfile(PLOTS_DIR, sprintf('clval_%s.png', sc.name));
    exportgraphics(fig, out_png, 'Resolution',150);
    close(fig);
    fprintf('  Plot: plots/clval_%s.png\n', sc.name);
end

%% 4. Summary table --------------------------------------------------------
valid = find(arrayfun(@(r) isfield(r,'rmse_tq') && ~isempty(r.rmse_tq), R));

fprintf('\n\n');
fprintf('=======================================================================\n');
fprintf('  CLOSED-LOOP VALIDATION SUMMARY — QAT LSTM-32 ROM\n');
fprintf('  (S-Function block replaces Combustion subsystem)\n');
fprintf('=======================================================================\n');
fprintf('%-30s  %11s  %9s  %8s  %8s\n', ...
    'Scenario', 'Tq RMSE N·m', 'R² Torque', 'Spd rpm', 'R² Speed');
fprintf('%s\n', repmat('-',1,76));
for k = valid
    fprintf('%-30s  %11.4f  %9.5f  %8.2f  %8.5f\n', ...
        R(k).name, R(k).rmse_tq, R(k).r2_tq, R(k).rmse_rpm, R(k).r2_spd);
end
fprintf('%s\n', repmat('-',1,76));
if ~isempty(valid)
    avg_rmse_tq  = mean([R(valid).rmse_tq]);
    avg_r2_tq    = mean([R(valid).r2_tq]);
    avg_rmse_rpm = mean([R(valid).rmse_rpm]);
    avg_r2_spd   = mean([R(valid).r2_spd]);
    fprintf('%-30s  %11.4f  %9.5f  %8.2f  %8.5f\n', ...
        'AVERAGE', avg_rmse_tq, avg_r2_tq, avg_rmse_rpm, avg_r2_spd);
end
fprintf('=======================================================================\n');

%% 5. Summary figure (2×2: Torque bar + Torque traces, Speed bar + Speed traces)
if numel(valid) >= 1
    fig = figure('Visible','off','Position',[50 50 1300 700]);
    cmap = lines(n_sc);
    sc_labels = arrayfun(@(k) strrep(R(k).name,'_',' '), valid, 'UniformOutput',false);

    % Top-left: Torque RMSE bar (PRIMARY)
    subplot(2,2,1);
    tq_rmse_vals = arrayfun(@(k) R(k).rmse_tq, valid);
    tq_r2_vals   = arrayfun(@(k) R(k).r2_tq,   valid);
    b1 = bar(categorical(sc_labels), tq_rmse_vals, 'FaceColor','flat');
    b1.CData = cmap(valid,:);
    ylabel('Torque RMSE (N·m)');
    title('ROM Torque RMSE per Scenario  [PRIMARY]');
    grid on; ylim([0 max(tq_rmse_vals)*1.30 + 0.01]);
    for bi = 1:numel(tq_rmse_vals)
        text(bi, tq_rmse_vals(bi) + max(tq_rmse_vals)*0.05, ...
             sprintf('R^2=%.4f', tq_r2_vals(bi)), ...
             'HorizontalAlignment','center','FontSize',8);
    end

    % Top-right: all torque traces overlaid (HiFi=solid, ROM=dashed)
    ax_tq = subplot(2,2,2);
    hold on;
    h_tq = gobjects(numel(valid),1);
    for ki = 1:numel(valid)
        k = valid(ki);
        col = cmap(k,:);
        plot(R(k).t_tq_h, R(k).tq_h, '-',  'Color',col, 'LineWidth',1.0);
        h_tq(ki) = plot(R(k).t_tq_r, R(k).tq_r, '--', 'Color',col, 'LineWidth',1.6, ...
             'DisplayName', strrep(R(k).name,'_',' '));
    end
    xlabel('Time (s)'); ylabel('Torque (N·m)');
    title({'All Scenarios: HiFi Combustion (solid) vs ROM (dashed)'; 'Torque — direct ROM output'});
    legend(h_tq, 'Location','best','FontSize',7); grid on; xlim([0 T_sim]);

    % Bottom-left: Speed RMSE bar (secondary – closed-loop)
    subplot(2,2,3);
    spd_rmse_vals = arrayfun(@(k) R(k).rmse_rpm, valid);
    spd_r2_vals   = arrayfun(@(k) R(k).r2_spd,   valid);
    b2 = bar(categorical(sc_labels), spd_rmse_vals, 'FaceColor','flat');
    b2.CData = cmap(valid,:);
    ylabel('Speed RMSE (rpm)');
    title('Closed-Loop Speed RMSE per Scenario  [secondary]');
    grid on; ylim([0 max(spd_rmse_vals)*1.30 + 0.1]);
    for bi = 1:numel(spd_rmse_vals)
        text(bi, spd_rmse_vals(bi) + max(spd_rmse_vals)*0.05, ...
             sprintf('R^2=%.4f', spd_r2_vals(bi)), ...
             'HorizontalAlignment','center','FontSize',8);
    end

    % Bottom-right: all speed traces overlaid
    ax_spd = subplot(2,2,4);
    hold on;
    h_spd = gobjects(numel(valid),1);
    for ki = 1:numel(valid)
        k = valid(ki);
        col = cmap(k,:);
        plot(R(k).t_h, R(k).spd_h, '-',  'Color',col, 'LineWidth',1.0);
        h_spd(ki) = plot(R(k).t_r, R(k).spd_r, '--', 'Color',col, 'LineWidth',1.6, ...
             'DisplayName', strrep(R(k).name,'_',' '));
    end
    xlabel('Time (s)'); ylabel('Engine Speed (rad/s)');
    title({'All Scenarios: HiFi (solid) vs ROM (dashed)'; 'Closed-Loop Speed — downstream check'});
    legend(h_spd, 'Location','best','FontSize',7); grid on; xlim([0 T_sim]);

    sgtitle('QAT LSTM-32 Closed-Loop Validation Summary', ...
            'FontSize',13,'FontWeight','bold');
    exportgraphics(fig, fullfile(PLOTS_DIR,'clval_summary.png'), 'Resolution',150);
    close(fig);
    fprintf('\nSummary plot: plots/clval_summary.png\n');
end

%% 6. Save metrics ---------------------------------------------------------
cl_metrics.scenarios = R;
cl_metrics.valid_idx = valid;
cl_metrics.T_sim     = T_sim;
cl_metrics.dt        = dt;
cl_metrics.hifi_mdl  = hifi_mdl;
cl_metrics.rom_mdl   = cc_mdl;
if ~isempty(valid)
    cl_metrics.avg_rmse_tq_nm    = avg_rmse_tq;
    cl_metrics.avg_r2_tq         = avg_r2_tq;
    cl_metrics.avg_rmse_spd_rpm  = avg_rmse_rpm;
    cl_metrics.avg_r2_spd        = avg_r2_spd;
end
save(fullfile(DATA_DIR,'cl_validation_results.mat'), 'cl_metrics');
fprintf('Metrics saved: data/cl_validation_results.mat\n');

% Close models
if bdIsLoaded(hifi_mdl), close_system(hifi_mdl, 0); end
if bdIsLoaded(cc_mdl),   close_system(cc_mdl,   0); end
warning('on','all');

fprintf('\n=== Closed-Loop Validation Complete ===\n');


%% ==========================================================================
%  Helper functions
%% ==========================================================================

function [t_out, y_out] = extract_signal(simOut, sig_name)
%EXTRACT_SIGNAL  Pull a logged signal from logsout.
    ls  = simOut.logsout;
    el  = ls.getElement(sig_name);
    t_out = el.Values.Time;
    y_out = squeeze(el.Values.Data);
end

function sig = build_step_signal(t_vec, levels, switch_times)
%BUILD_STEP_SIGNAL  Piecewise constant signal on t_vec.
%  levels(1) active for t < switch_times(2), etc.
    sig = zeros(size(t_vec));
    for k = 1:numel(switch_times)
        if k < numel(switch_times)
            idx = t_vec >= switch_times(k) & t_vec < switch_times(k+1);
        else
            idx = t_vec >= switch_times(k);
        end
        sig(idx) = levels(k);
    end
end
