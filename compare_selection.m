outdir = "runs/selection_accuracy";
mkdir(outdir);

% use val accuracy for all six, then test
disp('use val acc');
hist1 = readtable("runs/scratch_main_original/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
cnnMainPick = hist1(best, :);
cnnMainPick.run_id = "scratch_main_original";
hist1 = readtable("runs/resnet18_main_original/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
res_mainPick = hist1(best, :);
res_mainPick.run_id = "resnet18_main_original";

hist1 = readtable("runs/scratch_matched_original/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
cnn_origPick = hist1(best, :);
cnn_origPick.run_id = "scratch_matched_original";

hist1 = readtable("runs/resnet18_matched_original/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
rnOrigPick = hist1(best, :);
rnOrigPick.run_id = "resnet18_matched_original";


hist1 = readtable("runs/scratch_matched_foreground/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
cnnFgPick = hist1(best, :);
cnnFgPick.run_id = "scratch_matched_foreground";

hist1 = readtable("runs/resnet18_matched_foreground/checkpoint_comparison.csv", 'TextType', 'string');
[~, best] = max(hist1.validation_accuracy);
rn_fgPick = hist1(best, :);
rn_fgPick.run_id = "resnet18_matched_foreground";

plan = [cnnMainPick; res_mainPick; ...
  cnn_origPick; rnOrigPick; ...
      cnnFgPick; rn_fgPick];
fout = fullfile(outdir, "selection_plan.csv");
writetable(plan, fout);
disp('picks saved');


disp("main test");
[imgs, rows] = read_images("test.csv", "original");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
load(cnnMainPick.checkpoint_file, 'net');
dest = fullfile(outdir, "scratch_main_original", "test");
cnn_main = evaluate_model(net, cnn_pic, rows, dest);
load(res_mainPick.checkpoint_file, 'net');
dest = fullfile(outdir, "resnet18_main_original", "test");
resnetMain = evaluate_model(net, imgs, rows, dest);
clear imgs cnn_pic;

fprintf('matched: original\n');
[imgs, rows] = read_images("robustness_test.csv", "original");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
load(cnn_origPick.checkpoint_file, 'net');
dest = fullfile(outdir, "scratch_matched_original", "test");
cnnOrig = evaluate_model(net, cnn_pic, rows, dest);
load(rnOrigPick.checkpoint_file, 'net');
dest = fullfile(outdir, "resnet18_matched_original", "test");
rn_original = evaluate_model(net, imgs, rows, dest);
clear imgs cnn_pic;

disp('FG test');
[imgs, rows] = read_images("robustness_test.csv", "foreground");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
load(cnnFgPick.checkpoint_file, 'net');
dest = fullfile(outdir, "scratch_matched_foreground", "test");
cnn_fg = evaluate_model(net, cnn_pic, rows, dest);
load(rn_fgPick.checkpoint_file, 'net');
dest = fullfile(outdir, "resnet18_matched_foreground", "test");
resFg = evaluate_model(net, imgs, rows, dest);
clear imgs cnn_pic;

summary = table();
summary.run_id = plan.run_id;
allScore = [cnn_main; resnetMain; cnnOrig; rn_original; ...
    cnn_fg; resFg];
summary = [summary allScore];
fout = fullfile(outdir, "selection_results.csv");
writetable(summary, fout);

% Background difference in percentage points.
cnn_accDiff = cnn_fg.accuracy - cnnOrig.accuracy;
cnn_accDiff = 100 * cnn_accDiff;
rn_accDiff = resFg.accuracy - rn_original.accuracy;
rn_accDiff = 100 * rn_accDiff;
cnn_f1diff = cnn_fg.macro_f1 - cnnOrig.macro_f1;
cnn_f1diff = 100 * cnn_f1diff;
rnF1diff = resFg.macro_f1 - rn_original.macro_f1;
rnF1diff = 100 * rnF1diff;
diffTab = table();
diffTab.model = ["scratch"; "resnet18"];
diffTab.accuracy_change_pp = [cnn_accDiff; rn_accDiff];
diffTab.macro_f1_change_pp = [cnn_f1diff; rnF1diff];
fout = fullfile(outdir, "background_changes.csv");
writetable(diffTab, fout);
disp(summary);
fprintf('files in %s\n', outdir);
