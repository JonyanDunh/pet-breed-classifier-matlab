outDir = "outputs";

[imgs, rows] = read_images("test.csv", "original");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
disp('test cnn, original');
load("runs/scratch_main_original/selected.mat", 'net');
oneOut = fullfile(outDir, "scratch_main_original", "test");
cnn_main = evaluate_model(net, cnn_pic, rows, oneOut);
disp("ResNet18 test");
load("runs/resnet18_main_original/selected.mat", 'net');
oneOut = fullfile(outDir, "resnet18_main_original", "test");
resnetMain = evaluate_model(net, imgs, rows, oneOut);
clear imgs cnn_pic;


[imgs, rows] = read_images("robustness_test.csv", "original");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
disp("cnn matched");
load("runs/scratch_matched_original/selected.mat", 'net');
oneOut = fullfile(outDir, "scratch_matched_original", "test");
cnnOrig = evaluate_model(net, cnn_pic, rows, oneOut);
disp('matched original - ResNet18');
load("runs/resnet18_matched_original/selected.mat", 'net');
oneOut = fullfile(outDir, "resnet18_matched_original", "test");
rn_original = evaluate_model(net, imgs, rows, oneOut);
clear imgs cnn_pic;

[imgs, rows] = read_images("robustness_test.csv", "foreground");
cnn_pic = single(imgs);
cnn_pic = cnn_pic / 255;
disp('foreground cnn');
load("runs/scratch_matched_foreground/selected.mat", 'net');
oneOut = fullfile(outDir, "scratch_matched_foreground", "test");
cnn_fg = evaluate_model(net, cnn_pic, rows, oneOut);

disp("ResNet18 / foreground");
load("runs/resnet18_matched_foreground/selected.mat", 'net');
oneOut = fullfile(outDir, "resnet18_matched_foreground", "test");
resFg = evaluate_model(net, imgs, rows, oneOut);
clear imgs cnn_pic;

mainTab = table();
mainTab.run_id = ["scratch_main_original"; "resnet18_main_original"];
mainScore = [cnn_main; resnetMain];
mainTab = [mainTab mainScore];
fout = fullfile(outDir, "main_results.csv");
writetable(mainTab, fout);
match_tab = table();
match_tab.run_id = ["scratch_matched_original"; "resnet18_matched_original"; ...
  "scratch_matched_foreground"; "resnet18_matched_foreground"];
matchScores = [cnnOrig; rn_original; cnn_fg; resFg];
match_tab = [match_tab matchScores];
fout = fullfile(outDir, "matched_results.csv");
writetable(match_tab, fout);
summary = [mainTab; match_tab];

% foreground minus original, in percentage points
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
fout = fullfile(outDir, "background_changes.csv");
writetable(diffTab, fout);
disp(summary);
fprintf('results: %s\n', outDir);
