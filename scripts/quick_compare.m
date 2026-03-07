%% quick_compare.m
% Run HiFi and ROM models with S2 scenario (rich transient, SA=15 deg)
% and produce a 2-panel comparison plot.

clc;
PROJ      = '/Users/arkadiyturevskiy/Documents/Claude/Claude_ROM_session4';
SRC       = fullfile(PROJ, 'src');
PLOTS_DIR = fullfile(PROJ, 'plots');
cd(PROJ);
addpath(SRC);
if ~isfolder(PLOTS_DIR), mkdir(PLOTS_DIR); end

hifi_mdl  = 'enginespeed_hifi_val';
rom_mdl   = 'enginespeed_qat_sfun';
hifi_file = fullfile(PROJ, [hifi_mdl '.slx']);
rom_file  = fullfile(PROJ, [rom_mdl  '.slx']);
assert(isfile(hifi_file), 'HiFi model not found');
assert(isfile(rom_file),  'ROM model not found');

warning('off','all');

%% Define scenario: rich multi-step throttle, SA=15 deg
T_sim = 25.0;
dt    = 0.05;
t_vec = (0:dt:T_sim)';

lvl = [5, 15, 30, 20, 45, 10, 38, 25, 50, 12, 40, 18, 35, 8, 22];
tsw = [0, 1.0, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.5, 18.0, 20.0, 21.5, 23.0, 24.5];
thr_sig = zeros(size(t_vec));
for k = numel(tsw):-1:1
    thr_sig(t_vec >= tsw(k)) = lvl(k);
end
sa_sig = 15 * ones(size(t_vec));

throttle_ts = [t_vec, thr_sig]; %#ok<NASGU>
sa_ts       = [t_vec, sa_sig];  %#ok<NASGU>
assignin('base', 'throttle_ts', throttle_ts);
assignin('base', 'sa_ts',       sa_ts);

%% Load and simulate
fprintf('Loading models…\n');
if ~bdIsLoaded(hifi_mdl), load_system(hifi_file); end
if ~bdIsLoaded(rom_mdl),  load_system(rom_file);  end

fprintf('Simulating HiFi… ');
simH = sim(hifi_mdl, 'StopTime', num2str(T_sim));
fprintf('done\n');

fprintf('Simulating ROM… ');
simR = sim(rom_mdl,  'StopTime', num2str(T_sim));
fprintf('done\n');

%% Extract signals
[t_sh, spd_h] = extract_sig(simH, 'Speed');
[t_th, tq_h]  = extract_sig(simH, 'Torque');
[t_sr, spd_r] = extract_sig(simR, 'Speed');
[t_tr, tq_r]  = extract_sig(simR, 'Torque');

%% Time-align on uniform 0.05s grid
t_min = max([t_sh(1), t_sr(1), t_th(1), t_tr(1)]);
t_max = min([t_sh(end), t_sr(end), t_th(end), t_tr(end)]);
mask  = (t_vec >= t_min) & (t_vec <= t_max);
tc    = t_vec(mask);

spd_h_u = interp1(t_sh, spd_h, tc, 'linear', 'extrap');
spd_r_u = interp1(t_sr, spd_r, tc, 'linear', 'extrap');
tq_h_u  = interp1(t_th, tq_h,  tc, 'linear', 'extrap');
tq_r_u  = interp1(t_tr, tq_r,  tc, 'linear', 'extrap');

spd_h_rpm = spd_h_u * 60/(2*pi);
spd_r_rpm = spd_r_u * 60/(2*pi);

%% Metrics
err_tq   = tq_h_u - tq_r_u;
rmse_tq  = sqrt(mean(err_tq.^2));
r2_tq    = 1 - sum(err_tq.^2)/max(sum((tq_h_u-mean(tq_h_u)).^2),1e-12);

err_spd   = spd_h_rpm - spd_r_rpm;
rmse_rpm  = sqrt(mean(err_spd.^2));
r2_spd    = 1 - sum(err_spd.^2)/max(sum((spd_h_rpm-mean(spd_h_rpm)).^2),1e-12);

fprintf('\nTorque RMSE = %.3f N·m   R² = %.5f\n', rmse_tq, r2_tq);
fprintf('Speed  RMSE = %.2f rpm   R² = %.5f\n', rmse_rpm, r2_spd);

%% Plot
fig = figure('Position',[100 100 900 600],'Color','w');

% Panel 1 – Torque
ax1 = subplot(2,1,1);
plot(tc, tq_h_u, 'b-',  'LineWidth', 1.5, 'DisplayName', 'HiFi'); hold on;
plot(tc, tq_r_u, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('QAT ROM (RMSE=%.2f N·m, R²=%.4f)', rmse_tq, r2_tq));
ylabel('Torque (N·m)', 'FontSize', 11);
title('Engine Torque – HiFi vs QAT ROM (closed-loop, rich transient SA=15°)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 9);
grid on; xlim([0 T_sim]);
set(ax1, 'FontSize', 10);

% Panel 2 – Speed
ax2 = subplot(2,1,2);
plot(tc, spd_h_rpm, 'b-',  'LineWidth', 1.5, 'DisplayName', 'HiFi'); hold on;
plot(tc, spd_r_rpm, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('QAT ROM (RMSE=%.1f rpm, R²=%.4f)', rmse_rpm, r2_spd));
ylabel('Engine Speed (rpm)', 'FontSize', 11);
xlabel('Time (s)', 'FontSize', 11);
title('Engine Speed – HiFi vs QAT ROM (closed-loop)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 9);
grid on; xlim([0 T_sim]);
set(ax2, 'FontSize', 10);

outfile = fullfile(PLOTS_DIR, 'quick_compare.png');
exportgraphics(fig, outfile, 'Resolution', 150);
fprintf('\nSaved: %s\n', outfile);

close_system(hifi_mdl, 0);
close_system(rom_mdl,  0);
warning('on','all');

%% Helper
function [t, y] = extract_sig(simOut, signame)
    ls = simOut.get('logsout');
    el = ls.getElement(signame);
    ts = el.Values;
    t  = ts.Time;
    y  = squeeze(double(ts.Data));
    if ~iscolumn(t), t = t(:); end
    if ~iscolumn(y), y = y(:); end
end
