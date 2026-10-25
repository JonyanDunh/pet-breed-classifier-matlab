function [imgs, rows] = read_images(csvFile, imgType)
disp("read " + csvFile);
listFile = fullfile("data/pets/manifests", csvFile);
rows = readtable(listFile, 'TextType', 'string');
if imgType == "original"
  paths = rows.original_relpath;
else
  paths = rows.foreground_relpath;
end

picNum = height(rows);
imgs = zeros(224, 224, 3, picNum, 'uint8');

for i = 1:picNum
    file1 = fullfile("data/pets", paths(i));
    pic = imread(file1);
    imgs(:, :, :, i) = pic;
end
fprintf('%d pics, %s\n', picNum, imgType);
end
