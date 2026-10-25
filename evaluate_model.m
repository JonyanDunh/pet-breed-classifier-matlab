function scores = evaluate_model(net, imgs, rows, folder)
started = tic;
[~, prob] = classify(net, imgs, ...
    'MiniBatchSize', 32, 'ExecutionEnvironment', 'gpu');
seconds = toc(started);

realLab = rows.class_index_0based;
n = height(rows);
predLab = zeros(n, 1);
conf = zeros(n, 1);
realProb = zeros(n, 1);
counts = zeros(37, 37);
for i = 1:n
  one_prob = prob(i, :);
  [topProb, breed] = max(one_prob);
  conf(i) = topProb;
  predLab(i) = breed - 1;
  realBreed = realLab(i) + 1;
  realProb(i) = prob(i, realBreed);
  counts(realBreed, breed) = counts(realBreed, breed) + 1;
end

support = zeros(37, 1);
precision = zeros(37, 1);
recall = zeros(37, 1);
f1 = zeros(37, 1);
correct = 0;
for breed = 1:37
    tp = counts(breed, breed);
    realCount = counts(breed, :);
    predCount = counts(:, breed);
    support(breed) = sum(realCount);
    predicted = sum(predCount);
    % no prediction for a breed: precision is 0
    if predicted > 0
        precision(breed) = tp / predicted;
    end
    recall(breed) = tp / support(breed);
    f1(breed) = 2 * tp / (support(breed) + predicted);
    correct = correct + tp;
end

scores = table();
scores.count = n;
scores.accuracy = correct / n;
scores.macro_precision = mean(precision);
scores.macro_recall = mean(recall);
scores.macro_f1 = mean(f1);
logProb = log(realProb);
scores.cross_entropy = -mean(logProb);
scores.inference_seconds = seconds;

% empty folder: only get scores
if folder ~= ""
    mkdir(folder);
    breedMap = readtable("data/pets/manifests/class_mapping.csv", 'TextType', 'string');
    realIdx = realLab + 1;
    predIdx = predLab + 1;
    predTab = table();
    predTab.image_id = rows.image_id;
    predTab.true_class_index_0based = realLab;
    predTab.pred_class_index_0based = predLab;
    predTab.true_breed = breedMap.breed(realIdx);
    predTab.pred_breed = breedMap.breed(predIdx);
    predTab.predicted_class_probability = conf;
    predFile = fullfile(folder, "predictions.csv");
    writetable(predTab, predFile);
    scoreFile = fullfile(folder, "metrics.csv");
    writetable(scores, scoreFile);
    classTab = table();
    classTab.class_index_0based = (0:36)';
    classTab.breed = breedMap.breed;
    classTab.support = support;
    classTab.precision = precision;
    classTab.recall = recall;
    classTab.f1 = f1;
    classCsv = fullfile(folder, "per_class.csv");
    writetable(classTab, classCsv);
    mapFile = fullfile(folder, "class_mapping.csv");
    writetable(breedMap, mapFile);


    norm_cm = zeros(37, 37);
    for breed = 1:37
        norm_cm(breed, :) = counts(breed, :) / support(breed);
    end
    cmFile = fullfile(folder, "confusion_counts.csv");
    writematrix(counts, cmFile);
    normFile = fullfile(folder, "confusion_row_normalised.csv");
    writematrix(norm_cm, normFile);

    breedText = strrep(breedMap.breed, "_", " ");
    for plot_no = 1:2
        picture = figure('Position', [100 100 1500 1300]);
        if plot_no == 1
            imagesc(counts);
            filename = "confusion_counts.png";
        else
            imagesc(norm_cm);
            filename = "confusion_row_normalised.png";
        end
        axis image;
        colorbar;
        xticks(1:37);
        yticks(1:37);
        xticklabels(breedText);
        yticklabels(breedText);
        xtickangle(90);
        xlabel('Predicted breed');
        ylabel('True breed');
        picFile = fullfile(folder, filename);
        saveas(picture, picFile);
        close(picture);
    end

    pairs = table();
    for predicted = 1:37
        for truth = 1:37
            if truth ~= predicted && counts(truth, predicted) > 0
                pair = table();
                pair.true_breed = breedMap.breed(truth);
                pair.pred_breed = breedMap.breed(predicted);
                pair.count = counts(truth, predicted);
                pairs = [pairs; pair];
            end
        end
    end
    pairs = sortrows(pairs, 'count', 'descend');
    pairsFile = fullfile(folder, "confused_pairs.csv");
    writetable(pairs, pairsFile);
    wrong = [];
    for i = 1:n
        if realLab(i) ~= predLab(i)
            wrong = [wrong; i];
        end
    end
    wrongProb = conf(wrong);
    [~, order] = sort(wrongProb, 'descend');
    first12 = order(1:12);
    examples = wrong(first12);
    errTab = predTab(examples, :);
    errFile = fullfile(folder, "error_examples.csv");
    writetable(errTab, errFile);

    picture = figure('Position', [100 100 1600 1100]);
    for i = 1:12
      row = examples(i);
      subplot(3, 4, i);
      pic1 = imgs(:, :, :, row);
      image(pic1);
      axis image off;
      realBreed = realIdx(row);
      predictedBreed = predIdx(row);
      realText = "True: " + breedText(realBreed);
      predText = "Pred: " + breedText(predictedBreed);
      imgId = rows.image_id(row);
      titleTxt = [imgId; realText; predText];
      title(titleTxt, 'Interpreter', 'none');
    end
    picFile = fullfile(folder, "error_examples.png");
    saveas(picture, picFile);
    close(picture);
    disp("saved: " + folder);
end
end
