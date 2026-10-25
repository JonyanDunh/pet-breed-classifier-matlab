trainCsv = "train.csv";
valid_list = "validation.csv";
imgKind = "original";
saveDir = "runs/resnet18_main_original";
epNum = 20;
rates = [0.0001 0.001];

disp('ResNet18 train');
% keep pixels at 0 to 255 here
[trainImgs, trRows] = read_images(trainCsv, imgKind);
[valid_pic, validRows] = read_images(valid_list, imgKind);
breedMap = readtable("data/pets/manifests/class_mapping.csv", 'TextType', 'string');
trLabel = categorical(trRows.class_index_0based, 0:36, breedMap.breed);
validLab = categorical(validRows.class_index_0based, 0:36, breedMap.breed);

trainN = height(trRows);
batchN = trainN / 32;
iterPerEp = floor(batchN);
mkdir(saveDir);
history = table();
runDirs = ["candidate_01", "candidate_02"];

for candidate = 1:2
    rng(42, 'twister');
    lastOut = classificationLayer('Name', 'breed_output', ...
        'Classes', breedMap.breed);
    baseNet = resnet18('Weights', 'imagenet');
    layers = layerGraph(baseNet);

    % last two layers: 37 breeds
    new_fc = fullyConnectedLayer(37, 'Name', 'fc1000');
    layers = replaceLayer(layers, 'fc1000', new_fc);
    oldLayers = baseNet.Layers;
    last1 = oldLayers(end);
    lastName = last1.Name;
    layers = replaceLayer(layers, lastName, lastOut);
    rate = rates(candidate);
    runName = runDirs(candidate);
    runDir = fullfile(saveDir, runName);
    checkDir = fullfile(runDir, "checkpoints");
    mkdir(checkDir);
    fprintf('rate: %g\n', rate);

    aug = imageDataAugmenter('RandXReflection', true, ...
        'RandRotation', [-10 10], 'FillValue', 128);
    train_ds = augmentedImageDatastore([224 224], trainImgs, trLabel, ...
        'DataAugmentation', aug);
    opt = trainingOptions('sgdm', ...
        'InitialLearnRate', rate, ...
        'Momentum', 0.9, ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', epNum, ...
        'L2Regularization', 0, ...
        'ResetInputNormalization', false, ...
        'BatchNormalizationStatistics', 'moving', ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {valid_pic, validLab}, ...
        'ValidationPatience', Inf, ...
        'ValidationFrequency', iterPerEp, ...
        'CheckpointPath', checkDir, ...
        'ExecutionEnvironment', 'gpu', ...
        'Verbose', true, ...
        'Plots', 'none');

    started = tic;
    [net, info] = trainNetwork(train_ds, layers, opt);
    trainTime = toc(started);
    fprintf('ResNet18 training: %.1f s\n', trainTime);
    save_file = fullfile(runDir, "training_info.mat");
    save(save_file, 'info');

    lossVal = zeros(epNum, 1);
    accVal = zeros(epNum, 1);
    f1Val = zeros(epNum, 1);
    files = dir(checkDir);
    for epoch = 1:epNum
      % Match the iteration number in the file name.
      iteration = epoch * iterPerEp;
      prefix = "net_checkpoint__" + iteration + "__";
      for file = 1:length(files)
          checkpoint = files(file);
          checkName = checkpoint.name;
          if startsWith(checkName, prefix)
              checkFile = fullfile(checkDir, checkName);
              break;
          end
      end
      load(checkFile, 'net');
      scores = evaluate_model(net, valid_pic, validRows, "");
      lossVal(epoch) = scores.cross_entropy;
      accVal(epoch) = scores.accuracy;
      f1Val(epoch) = scores.macro_f1;
      fprintf('val %d: acc=%.3f, F1=%.3f\n', epoch, scores.accuracy, scores.macro_f1);
      entry = table();
      entry.candidate = candidate;
      entry.learning_rate = rate;
      entry.epoch = epoch;
      entry.iteration = iteration;
      entry.validation_loss = scores.cross_entropy;
      entry.validation_accuracy = scores.accuracy;
      entry.validation_macro_f1 = scores.macro_f1;
      entry.training_seconds = trainTime;
      entry.checkpoint_file = checkFile;
      history = [history; entry];
    end


    % Use epochs on the x axis.
    batchCount = length(info.TrainingLoss);
    batchIdx = 1:batchCount;
    x = batchIdx / iterPerEp;
    curveFig = figure('Position', [100 100 1200 360]);
    subplot(1, 3, 1);
    plot(x, info.TrainingLoss);
    hold on;
    plot(1:epNum, lossVal, 'o-');
    xlabel('Epoch');
    ylabel('Cross entropy');
    legend('Training batches', 'Validation');
    subplot(1, 3, 2);
    trainAcc = info.TrainingAccuracy / 100;
    plot(x, trainAcc);
    hold on;
    plot(1:epNum, accVal, 'o-');
    xlabel('Epoch');
    ylabel('Accuracy');
    ylim([0 1]);

    subplot(1, 3, 3);
    plot(1:epNum, f1Val, 'o-');
    xlabel('Epoch');
    ylabel('Validation macro-F1');
    ylim([0 1]);
    save_file = fullfile(runDir, "learning_curves.png");
    saveas(curveFig, save_file);
    close(curveFig);
end

% Pick the first row when F1 is equal.
save_file = fullfile(saveDir, "checkpoint_comparison.csv");
writetable(history, save_file);
[~, best] = max(history.validation_macro_f1);
selection = history(best, :);
load(selection.checkpoint_file, 'net');
save_file = fullfile(saveDir, "selected.mat");
save(save_file, 'net', 'selection');

save_file = fullfile(saveDir, "selection.csv");
writetable(selection, save_file);
validDir = fullfile(saveDir, "validation");
scores = evaluate_model(net, valid_pic, validRows, validDir);
fprintf('best epoch: %d / lr=%g\n', selection.epoch, selection.learning_rate);
disp("saved at " + saveDir);
