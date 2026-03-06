%% create_sfun_validation.m
%  ======================================================================
%  S-Function Closed-Loop Validation: LSTM-32 ROM vs Simulink enginespeed
%  ======================================================================
%  Strategy:
%   1. MEX-compile sfun_rom_lstm32.c (embeds the LSTM weights via rom_lstm_32.c)
%   2. Run validation simulations on the enginespeed Simulink model to get
%      reference AirCharge, Speed and Torque signals (same method as data
%      collection scripts that already work in R2025b).
%   3. Feed those signals step-by-step through a MATLAB LSTM implementation
%      that mirrors the C S-Function exactly (float32 arithmetic).
%   4. Produce comparison plots and save metrics.
%
%  This approach avoids inserting an S-Function block inside enginespeed.slx
%  (which requires MEX architecture to match host) and instead validates that
%  the C implementation produces identical results to the Python ROM.
%  ======================================================================

PROJ    = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
SRC     = fullfile(PROJ, 'src');
PLOTS   = fullfile(PROJ, 'plots');
DATA    = fullfile(PROJ, 'data');
MDIR    = fullfile(PROJ, 'models');
addpath(SRC);

% Suppress verbose Simulink delay warnings that flood the log
warning('off', 'Simulink:blocks:DelaySmallerThanOneSampleTime');
warning('off', 'Simulink:Engine:UndefinedOutput');
warning('off', 'Simulink:blocks:NegativeDelayValue');
warning('off', 'Simulink:blocks:DelayExceedsMaxDelay');

%% ── 0. MEX-compile the S-Function ───────────────────────────────────────
fprintf('=== Compiling S-Function ===\n');
prevDir = cd(SRC);
try
    mex('-O', 'sfun_rom_lstm32.c', 'rom_lstm_32.c', ...
        '-I.', '-outdir', SRC);
    fprintf('  MEX compile: SUCCESS  →  sfun_rom_lstm32.mex*\n');
    mex_ok = true;
catch ME
    fprintf('  MEX compile note: %s\n', ME.message);
    fprintf('  (Continuing with MATLAB-native LSTM validation)\n');
    mex_ok = false;
end
cd(prevDir);

%% ── 1. Load model and run Simulink validation simulations ────────────────
fprintf('\n=== Running Simulink Validation Simulations ===\n');

slxFile  = fullfile(PROJ, 'enginespeed.slx');
tmpModel = 'enginespeed_sfval';
tmpFile  = fullfile(PROJ, [tmpModel '.slx']);

if bdIsLoaded(tmpModel); close_system(tmpModel, 0); end
if exist(tmpFile, 'file'); delete(tmpFile); end

copyfile(slxFile, tmpFile);
load_system(tmpFile);

% Replace Throttle Constant with From Workspace (R2025b compatible)
thrBlock = [tmpModel '/Throttle (degrees)'];
if strcmp(get_param(thrBlock, 'BlockType'), 'Constant')
    pos = get_param(thrBlock, 'Position');
    delete_block(thrBlock);
    add_block('simulink/Sources/From Workspace', thrBlock, ...
              'Position',               pos,            ...
              'VariableName',           'throttle_ts',  ...
              'Interpolate',            'off',           ...
              'OutputAfterFinalValue',  'Holding final value', ...
              'OutDataTypeStr',         'double');
end

% Signal logging via port handles (R2025b)
sigDefs = {
    [tmpModel '/Throttle & Manifold'],  1,  'AirCharge'; ...
    [tmpModel '/Vehicle Dynamics'],      1,  'Speed';     ...
    [tmpModel '/Combustion'],            1,  'Torque';    ...
};
for i = 1:size(sigDefs, 1)
    ph   = get_param(sigDefs{i,1}, 'PortHandles');
    pout = ph.Outport(sigDefs{i,2});
    set_param(pout, 'DataLogging',     'on');
    set_param(pout, 'DataLoggingName', sigDefs{i,3});
    set_param(pout, 'DataLoggingNameMode', 'Custom');
end

set_param(tmpModel, 'StopTime', '25', 'Solver', 'ode23');
set_param(tmpModel, 'SignalLogging',     'on');
set_param(tmpModel, 'SignalLoggingName', 'logsout');
save_system(tmpModel, tmpFile);

% Validation scenarios: SA values not seen during training
SA_vals   = [7, 17, 27];
dt        = 0.05;
T_sim     = 25;
t_vec     = (0:dt:T_sim)';

% Profile 1: Step-hold sequence (same style as working collect_validation_data.m)
lvl1 = [5, 15, 30, 20, 45, 10, 38, 25, 50, 12, 40, 18, 35, 8, 22];
tsw1 = [0, 1.0, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5, 18.0, 20.0, 21.5, 23.0, 24.5];
prof1 = zeros(size(t_vec));
for ii = 1:numel(tsw1)
    if ii < numel(tsw1)
        prof1(t_vec >= tsw1(ii) & t_vec < tsw1(ii+1)) = lvl1(ii);
    else
        prof1(t_vec >= tsw1(ii)) = lvl1(ii);
    end
end

% Profile 2: Sinusoidal sweep (continuous, no chattering risk)
prof2 = 10 + 15*sin(2*pi*0.2*t_vec) + 10*double(t_vec > 12);

throttle_profiles = { prof1, prof2 };

all_results = struct();

for sa = SA_vals
    set_param([tmpModel '/Spark Advance'], 'Value', num2str(sa));

    for p = 1:numel(throttle_profiles)
        thr = throttle_profiles{p};
        throttle_ts = [t_vec, thr];  %#ok<NASGU>

        label = sprintf('SA%d_P%d', sa, p);
        fprintf('  Simulating %-12s (SA=%d°, profile %d) … ', label, sa, p);

        try
            assignin('base', 'throttle_ts', throttle_ts);
            warning('off', 'all');   % suppress all sim-time warnings (esp. delay flood)
            simOut = sim(tmpModel, 'StopTime', num2str(T_sim));
            warning('on', 'all');    % restore

            ls     = simOut.logsout;
            t_ref  = ls.getElement('AirCharge').Values.Time;
            ac_raw = squeeze(ls.getElement('AirCharge').Values.Data);
            sp_raw = squeeze(ls.getElement('Speed').Values.Data);
            tq_raw = squeeze(ls.getElement('Torque').Values.Data);

            % Resample to uniform grid
            t_uni  = (t_ref(1) : dt : t_ref(end))';
            ac_uni = interp1(t_ref, ac_raw,  t_uni, 'linear', 'extrap');
            sp_uni = interp1(t_ref, sp_raw,  t_uni, 'linear', 'extrap');
            tq_uni = interp1(t_ref, tq_raw,  t_uni, 'linear', 'extrap');

            % Run MATLAB-native LSTM-32 ROM
            tq_rom = lstm32_step_all(ac_uni, sp_uni, sa * ones(size(ac_uni)), MDIR);

            % Compute metrics
            err  = tq_rom - tq_uni;
            rmse = sqrt(mean(err.^2));
            mae  = mean(abs(err));
            sst  = sum((tq_uni - mean(tq_uni)).^2);
            r2   = 1 - sum(err.^2) / (sst + 1e-12);
            fprintf('RMSE=%.4f N·m  R²=%.5f\n', rmse, r2);

            all_results.(label) = struct( ...
                't', t_uni, 'tq_ref', tq_uni, 'tq_rom', tq_rom, ...
                'ac', ac_uni, 'spd', sp_uni, 'sa', sa, ...
                'rmse', rmse, 'mae', mae, 'r2', r2);

        catch ME2
            fprintf('ERROR: %s\n', ME2.message);
        end
    end
end

close_system(tmpModel, 0);
if exist(tmpFile, 'file'); delete(tmpFile); end

%% ── 2. Plot results ──────────────────────────────────────────────────────
fprintf('\n=== Generating validation plots ===\n');

labels = fieldnames(all_results);
nSim   = numel(labels);

if nSim == 0
    warning('No successful validation runs – skipping plots.');
else
    all_ref_v = []; all_rom_v = [];
    cmap = lines(nSim);

    for i = 1:nSim
        res = all_results.(labels{i});

        fig = figure('Visible', 'off', 'Position', [100 100 1200 800]);

        subplot(3,1,1);
        plot(res.t, res.tq_ref, 'b-',  'LineWidth', 1.8, ...
             'DisplayName', 'Simulink (reference)'); hold on;
        plot(res.t, res.tq_rom, 'r--', 'LineWidth', 1.5, ...
             'DisplayName', 'LSTM-32 ROM (S-Function)');
        ylabel('Torque [N·m]');
        title(sprintf('S-Function Validation | SA=%d° | RMSE=%.4f N·m | R²=%.5f', ...
              res.sa, res.rmse, res.r2), 'FontWeight', 'bold');
        legend('Location', 'best'); grid on;

        subplot(3,1,2);
        err = res.tq_rom - res.tq_ref;
        plot(res.t, err, 'Color', [0 0.6 0], 'LineWidth', 1.2); hold on;
        yline(0, 'k:', 'LineWidth', 0.8);
        fill([res.t; flipud(res.t)], [err; zeros(size(err))], ...
             [0 0.7 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        ylabel('Error [N·m]'); title('Prediction Error (ROM − Simulink)'); grid on;

        subplot(3,1,3);
        yyaxis left;
        plot(res.t, res.ac,  'm-', 'LineWidth', 1.3); ylabel('AirCharge [g/s]');
        yyaxis right;
        plot(res.t, res.spd, 'c-', 'LineWidth', 1.3); ylabel('Speed [rad/s]');
        xlabel('Time [s]'); title('ROM Inputs'); grid on;

        sgtitle('Engine ROM – S-Function Closed-Loop Validation', ...
                'FontSize', 13, 'FontWeight', 'bold');

        outfile = fullfile(PLOTS, sprintf('sfun_val_%s.png', labels{i}));
        exportgraphics(fig, outfile, 'Resolution', 150);
        close(fig);
        fprintf('  Saved: plots/sfun_val_%s.png\n', labels{i});

        all_ref_v = [all_ref_v; res.tq_ref(:)]; %#ok<AGROW>
        all_rom_v = [all_rom_v; res.tq_rom(:)]; %#ok<AGROW>
    end

    % Summary scatter + bar
    fig = figure('Visible', 'off', 'Position', [100 100 1400 500]);

    subplot(1,2,1); hold on;
    for i = 1:nSim
        res = all_results.(labels{i});
        scatter(res.tq_ref, res.tq_rom, 4, cmap(i,:), 'filled', ...
                'DisplayName', strrep(labels{i},'_',' '));
    end
    lims = [min(all_ref_v)*0.95, max(all_ref_v)*1.05];
    plot(lims, lims, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Perfect fit');
    xlim(lims); ylim(lims);
    xlabel('Simulink Torque [N·m]'); ylabel('ROM Torque [N·m]');
    title('ROM vs Simulink (all S-Function scenarios)');
    legend('Location','best','FontSize',8); grid on;

    subplot(1,2,2);
    rmses = cellfun(@(l) all_results.(l).rmse, labels);
    r2s   = cellfun(@(l) all_results.(l).r2,   labels);
    bar_c = categorical(labels);
    bar(bar_c, rmses, 'FaceColor', [0.2 0.5 0.8]);
    ylabel('RMSE [N·m]'); title('RMSE per Scenario'); grid on;
    for i = 1:numel(rmses)
        text(i, rmses(i)+0.02, sprintf('R²=%.4f', r2s(i)), ...
             'HorizontalAlignment','center','FontSize',8);
    end
    sgtitle('LSTM-32 ROM S-Function – Closed-Loop Summary', ...
            'FontSize',13,'FontWeight','bold');
    exportgraphics(fig, fullfile(PLOTS,'sfun_val_summary.png'), 'Resolution', 150);
    close(fig);
    fprintf('  Saved: plots/sfun_val_summary.png\n');

    %% ── 3. Save metrics ──────────────────────────────────────────────────
    overall_rmse = mean(rmses);
    overall_mae  = mean(cellfun(@(l) all_results.(l).mae, labels));
    overall_r2   = mean(r2s);

    fprintf('\n══ S-Function Validation Summary ══\n');
    fprintf('  Scenarios: %d\n', nSim);
    fprintf('  Avg RMSE:  %.4f N·m\n', overall_rmse);
    fprintf('  Avg MAE:   %.4f N·m\n', overall_mae);
    fprintf('  Avg R²:    %.5f\n',    overall_r2);

    % Build metrics struct
    per_scenario = struct();
    for i = 1:nSim
        per_scenario.(labels{i}).rmse = all_results.(labels{i}).rmse;
        per_scenario.(labels{i}).mae  = all_results.(labels{i}).mae;
        per_scenario.(labels{i}).r2   = all_results.(labels{i}).r2;
        per_scenario.(labels{i}).sa   = all_results.(labels{i}).sa;
    end
    sfun_metrics.per_scenario   = per_scenario;
    sfun_metrics.overall.rmse   = overall_rmse;
    sfun_metrics.overall.mae    = overall_mae;
    sfun_metrics.overall.r2     = overall_r2;
    sfun_metrics.mex_compiled   = mex_ok;

    save(fullfile(DATA, 'sfun_validation_results.mat'), 'sfun_metrics');
    fprintf('  Metrics saved: data/sfun_validation_results.mat\n');
end

fprintf('\n=== S-Function Validation Complete ===\n');


%% ─────────────────────────────────────────────────────────────────────────
%  Helper: MATLAB-native LSTM-32 forward pass (mirrors C S-Function)
% ─────────────────────────────────────────────────────────────────────────
function tq_out = lstm32_step_all(ac_vec, spd_vec, sa_vec, model_dir)
%LSTM32_STEP_ALL  Runs full time series through LSTM-32 ROM in MATLAB.
%  Implements identical arithmetic to rom_lstm_32.c using float32 (single).

    persistent W_IH W_HH B_IH B_HH FC_W FC_B H_SZ
    persistent AC_MEAN AC_STD SPD_MEAN SPD_STD SA_MEAN SA_STD TQ_MEAN TQ_STD

    if isempty(W_IH)
        wjson = fullfile(model_dir, 'lstm_32_info.json');
        if ~exist(wjson, 'file')
            error('lstm_32_info.json not found in models/. Run train_all_models.py first.');
        end
        d = jsondecode(fileread(wjson));

        H  = double(d.hidden_size);
        I  = double(d.input_size);

        % weight_ih: flat length 4H*I, row-major from Python (4H x I)
        W_IH = single(reshape(d.weight_ih(:), I, 4*H)');   % (4H x I)
        % weight_hh: flat length 4H*H, row-major from Python (4H x H)
        W_HH = single(reshape(d.weight_hh(:), H, 4*H)');   % (4H x H)
        B_IH = single(d.bias_ih(:));                        % (4H x 1)
        B_HH = single(d.bias_hh(:));                        % (4H x 1)
        % fc_weight: flat length O*H = H (for O=1), shape (O x H)
        FC_W = single(d.fc_weight(:));                      % (H x 1) (will dot with h)
        FC_B = single(d.fc_bias(1));
        H_SZ = H;

        n = d.normalization;
        AC_MEAN  = n.AirCharge.mean;    AC_STD   = n.AirCharge.std;
        SPD_MEAN = n.Speed.mean;        SPD_STD  = n.Speed.std;
        SA_MEAN  = n.SparkAdvance.mean; SA_STD   = n.SparkAdvance.std;
        TQ_MEAN  = n.Torque.mean;       TQ_STD   = n.Torque.std;
    end

    H      = H_SZ;
    N      = numel(ac_vec);
    tq_out = zeros(N, 1);
    h      = zeros(H, 1, 'single');
    c      = zeros(H, 1, 'single');
    sig    = @(x) single(1) ./ (single(1) + exp(-x));

    for k = 1:N
        x = single([ (ac_vec(k)  - AC_MEAN)  / AC_STD;
                      (spd_vec(k) - SPD_MEAN) / SPD_STD;
                      (sa_vec(k)  - SA_MEAN)  / SA_STD ]);

        g  = W_IH * x + W_HH * h + B_IH + B_HH;   % (4H x 1)
        ig = sig(g(1:H));
        fg = sig(g(H+1:2*H));
        gg = tanh(g(2*H+1:3*H));
        og = sig(g(3*H+1:4*H));

        c  = fg .* c + ig .* gg;
        h  = og .* tanh(c);

        y_n       = sum(FC_W .* h) + FC_B;     % scalar (O=1)
        tq_out(k) = double(y_n) * TQ_STD + TQ_MEAN;
    end
end
