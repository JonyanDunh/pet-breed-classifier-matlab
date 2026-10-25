train_list = "train.csv";
valCsv = "validation.csv";
imgType = "original";
outdir = "runs/scratch_main_original";
epoch_n = 20;
lr_list = [0.01 0.001];

disp("cnn " + imgType);
[train_pic, train_tab] = read_images(train_list, imgType);
[valImgs, val_tab] = read_images(valCsv, imgType);
breeds = readtable("data/pets/manifests/class_mapping.csv", 'TextType', 'string');
trainLab = categorical(train_tab.class_index_0based, 0:36, breeds.breed);
valLabel = categorical(val_tab.class_index_0based, 0:36, breeds.breed);

% CNN input: 0 to 1
train_pic = single(train_pic);
train_pic = train_pic / 255;
valImgs = single(valImgs);
valImgs = valImgs / 255;

nTrain = height(train_tab);
batch_num = nTrain / 32;
iterEpoch = floor(batch_num);
mkdir(outdir);
history = table();
try_dirs = ["candidate_01", "candidate_02"];


for candidate = 1:2
  rng(42, 'twister');
  outLayer = classificationLayer('Name', 'breed_output', ...
      'Classes', breeds.breed);
  layers = [
      imageInputLayer([224 224 3], 'Normalization', 'none')
      convolution2dLayer(3, 8, 'Padding', 'same')
      batchNormalizationLayer
      reluLayer
      maxPooling2dLayer(2, 'Stride', 2)
      convolution2dLayer(3, 16, 'Padding', 'same')
      batchNormalizationLayer
      reluLayer
      maxPooling2dLayer(2, 'Stride', 2)

      convolution2dLayer(3, 32, 'Padding', 'same')
      batchNormalizationLayer
      reluLayer
      fullyConnectedLayer(37)
      softmaxLayer
      outLayer];
  lr = lr_list(candidate);
  tryName = try_dirs(candidate);
  one_dir = fullfile(outdir, tryName);
  ckpt_dir = fullfile(one_dir, "checkpoints");
  mkdir(ckpt_dir);
  fprintf('cnn lr=%g\n', lr);

  aug = imageDataAugmenter('RandXReflection', true, ...
      'RandRotation', [-10 10], 'FillValue', 128 / 255);
  trainData = augmentedImageDatastore([224 224], train_pic, trainLab, ...
      'DataAugmentation', aug);
  opts = trainingOptions('sgdm', ...
      'InitialLearnRate', lr, ...
      'Momentum', 0.9, ...
      'MiniBatchSize', 32, ...
      'MaxEpochs', epoch_n, ...
      'L2Regularization', 0, ...
      'ResetInputNormalization', false, ...
      'BatchNormalizationStatistics', 'moving', ...
      'Shuffle', 'every-epoch', ...
      'ValidationData', {valImgs, valLabel}, ...
      'ValidationPatience', Inf, ...
      'ValidationFrequency', iterEpoch, ...
      'CheckpointPath', ckpt_dir, ...
      'ExecutionEnvironment', 'gpu', ...
      'Verbose', true, ...
      'Plots', 'none');

  started = tic;
  [net, info] = trainNetwork(trainData, layers, opts);
  train_sec = toc(started);
  fprintf('train took %.1f sec\n', train_sec);

  fout = fullfile(one_dir, "training_info.mat");
  save(fout, 'info');

  val_loss = zeros(epoch_n, 1);
  valAcc = zeros(epoch_n, 1);
  valF1 = zeros(epoch_n, 1);
  files = dir(ckpt_dir);
  for epoch = 1:epoch_n
      % checkpoint file has the iteration number
      iteration = epoch * iterEpoch;
      prefix = "net_checkpoint__" + iteration + "__";
      for file = 1:length(files)
          checkpoint = files(file);
          ckpt_name = checkpoint.name;
          if startsWith(ckpt_name, prefix)
              ckptFile = fullfile(ckpt_dir, ckpt_name);
              break;
          end
      end
      load(ckptFile, 'net');
      scores = evaluate_model(net, valImgs, val_tab, "");
      val_loss(epoch) = scores.cross_entropy;
      valAcc(epoch) = scores.accuracy;
      valF1(epoch) = scores.macro_f1;
      fprintf('ep %d  F1 %.4f\n', epoch, scores.macro_f1);

      entry = table();
      entry.candidate = candidate;
      entry.learning_rate = lr;
      entry.epoch = epoch;
      entry.iteration = iteration;
      entry.validation_loss = scores.cross_entropy;
      entry.validation_accuracy = scores.accuracy;
      entry.validation_macro_f1 = scores.macro_f1;
      entry.training_seconds = train_sec;
      entry.checkpoint_file = ckptFile;
      history = [history; entry];
  end

  % Use epochs on the x axis
  nBatch = length(info.TrainingLoss);
  batch_x = 1:nBatch;
  x = batch_x / iterEpoch;
  fig1 = figure('Position', [100 100 1200 360]);
  subplot(1, 3, 1);
  plot(x, info.TrainingLoss);
  hold on;
  plot(1:epoch_n, val_loss, 'o-');
  xlabel('Epoch');
  ylabel('Cross entropy');
  legend('Training batches', 'Validation');

  subplot(1, 3, 2);
  trainAcc = info.TrainingAccuracy / 100;
  plot(x, trainAcc);
  hold on;
  plot(1:epoch_n, valAcc, 'o-');
  xlabel('Epoch');
  ylabel('Accuracy');
  ylim([0 1]);
  subplot(1, 3, 3);
  plot(1:epoch_n, valF1, 'o-');
  xlabel('Epoch');
  ylabel('Validation macro-F1');
  ylim([0 1]);
  fout = fullfile(one_dir, "learning_curves.png");
  saveas(fig1, fout);
  close(fig1);
end

% same F1: first row
fout = fullfile(outdir, "checkpoint_comparison.csv");
writetable(history, fout);

[~, best] = max(history.validation_macro_f1);
selection = history(best, :);
load(selection.checkpoint_file, 'net');
fout = fullfile(outdir, "selected.mat");
save(fout, 'net', 'selection');
fout = fullfile(outdir, "selection.csv");
writetable(selection, fout);
val_dir = fullfile(outdir, "validation");
scores = evaluate_model(net, valImgs, val_tab, val_dir);
fprintf('keep epoch %d, lr %g\n', selection.epoch, selection.learning_rate);
disp(outdir);
