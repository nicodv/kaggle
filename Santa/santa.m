function santa()
%% Packing Santa's Sleigh Kaggle Competition
% author: Nico de Vos
% 
% Initial analysis shows:
% X, Y and Z sizes range from 2 to 250, mean is 50, std. dev. is 62
% smallest box is 2x2x2, largest 247x249x250

%% Settings
width = 1000;
length = 1000;
% Calculate benchmark?
benchmark = 1;

%% Importing Data
presents = load('presents.mat');
presents = presents.presents;

% Save the columns as separate arrays, representing the IDs, widths,
% lengths, heights and orientation of each present
ID = presents(:,1);
width = presents(:,2);
length = presents(:,3);
height = presents(:,4);
% Note: there are 6 possible orientations, set to 1, meaning the original one
% 1 = WLH, width * length * height
% 2 = WHL
% 3 = HLW
% 4 = LWH
% 5 = HWL
% 6 = LHW
orient = ones(size(ID));

nPresents = size(presents, 1);

%% Benchmark
if benchmark == 1;
    % One possible approach would be to place the boxes in order going from top
    % to bottom.  This will have the advantage of having 0 penalty for the
    % ordering part of the metric, however the total height will likely not be
    % very good.  To do this, we'll start by filling presents along the
    % x-direction.  Once there's no more room in the x-direction, we increment
    % the y-direction.  Once there's no more room in the y-direction, we
    % increment the z-direction. (Note that this approach
    % takes 10 seconds or so to execute. Your own solution may take more or
    % less time, depending on the complexity and your machine.)

    xs = 1; ys = 1; zs = -1; % Initial coordinates for placing boxes
    lastRowIdxs = zeros(100,1); % Buffer for storing row indices
    lastLayerIdxs = zeros(500,1); % Buffer for storing layer indices
    numInRow = 0;
    numInLayer = 0;
    % ID and 8 sets of coordinates per present
    coords = zeros(nPresents,25);

    for i = 1:nPresents
        % Move to the next row if there isn't room
        if xs + width(i) > width + 1 % exceeded allowable width
            % increment y to ensure no overlap
            ys = ys + max(length(lastRowIdxs(1:numInRow)));
            xs = 1;
            numInRow = 0;
        end
        % Move to the next layer if there isn't room
        if ys + length(i) > length + 1 % exceeded allowable length
            % increment z to ensure no overlap
            zs = zs - max(height(lastLayerIdxs(1:numInLayer)));
            xs = 1;
            ys = 1;
            numInLayer = 0;
        end
        
        % Fill present coordinate matrix
        coords(i,1) = ID(i);
        coords(i,[2 8 14 20]) = xs;
        coords(i,[5 11 17 23]) = xs + width(i) - 1;
        coords(i,[3 6 15 18]) = ys;
        coords(i,[9 12 21 24]) = ys + length(i) - 1;
        coords(i,[4 7 10 13]) = zs;
        coords(i,[16 19 22 25]) = zs - height(i) + 1;

        % Update location info
        xs = xs + width(i);
        numInRow = numInRow+1;
        numInLayer = numInLayer+1;
        lastRowIdxs(numInRow) = ID(i);
        lastLayerIdxs(numInLayer) = ID(i);
    end

    % We started at z = -1 and went downward, need to shift so all z-values >= 1
    zCoords = coords(:,4:3:end);
    minZ = min(zCoords(:));
    coords(:,4:3:end) = zCoords - minZ + 1;
    benchmarkScore = evaluate(coords);
end

%% Initialization
% What follows is a smart initialization procedure that uses the correct order
% of presents, and encourages a strategy that is better in the long term by:
% - minimizing the average height (obvious)
% - minimizing the standard deviation of the height (i.e., the surface on which
%   to place new presents is as flat/smooth as possible)
% - minimizing free present surface (i.e., the presents are placed against each
%   other with as few gaps as possible)
%
% Initialization is done only once, so we're going to approach it a bit brute-force.

% function for determining the sum of free box surfaces
surface = @(H) sum(reshape(diff(H, 1, 1),1,[])) + ...
    sum(reshape(diff(H, 1, 2),1,[])) + numel(size(H));

orientOrder = zeros(nPresents, 6);

% matrix with heights per column
hMat = zeros(width, length);

% matrix with total gaps per column
gMat = zeros(width, length);

for i = 1:nPresents
    % determine preferred order of orientation: we prefer to place
    % presents as flat as possible (but no preference in x-y dimensions)
    % 1 = WLH, width * length * height
    % 2 = LWH
    % 3 = WHL
    % 4 = HWL
    % 5 = LHW
    % 6 = HLW
    minHLW = min(height(i), length(i), width(i));
    minHL = min(height(i), length(i));
    minHW = min(height(i), width(i));
    minLW = min(length(i), width(i));
    if height(i) == minHLW
        if length(i) == minLW
            orientOrder(i,:) = [1 2 3 4 5 6];
        else
            orientOrders(i,:) = [1 2 5 6 3 4];
    else if length(i) == minHLW
        if height(i) == minHW
            orientOrder(i,:) = [3 4 1 2 5 6];
        else
            orientOrders(i,:) = [3 4 5 6 1 2];
    else if width(i) == minHLW
        if length(i) == minHL
            orientOrder(i,:) = [5 6 3 4 1 2];
        else
            orientOrders(i,:) = [5 6 1 2 3 4];
    end
    
    % we also prefer a present placed against either another present, or on the
    % borders of the sleigh (this reduces complexity, hopefully still smart enough)
    potentialCoords = 

end

meanH = mean(hMat(:));
varH = var(hMat(:));
surfaceH = surface(hMat);


%% Statistics and plots regarding initialization

end


function metric = evaluate(coords)
% Compute evaluation metric that expresses how well Santa's sleigh is
% packed, judged by the overall compactness of the packing and the
% ordering of the presents: 
% metric  = 2 * max(z-coordinates) + sigma(order)

% Ideal order is the original order
idealOrder = [1:1e6]';

nPresents = size(coords, 1);

% Determine the max z-coordinate; this is the max height of the sleigh
maxZ = max(max(coords(:,4:3:end)));

% Go down the layers from top to bottom, reorder presents
% in numeric order for each layer
maxZCoord = zeros(nPresents,2);
for i = 1:nPresents
    maxZCoord(i,1) = coords(i);
    maxZCoord(i,2) = max(coords(i,4:3:end));
end
%sort max z-coord for each present
maxzCoordSorted = sortrows(maxZCoord,[-2 1]);
reOrder = maxzCoordSorted(:,1);

% Compare the new order to the ideal order
order = sum(abs(idealOrder - reOrder));

% Finally compute metric
metric = 2 * maxZ + order;

end


function fillrate(coords)
% Calculate the fill rate of the sleigh
% Determine the max coordinates
maxX = max(max(coords(:,2:3:end-2)));
maxY = max(max(coords(:,3:3:end-1)));
maxZ = max(max(coords(:,4:3:end)));

end


function createsubmission(filename, coords)
% Use fprintf to write the header, present IDs, and coordinates to a CSV file.
fileID = fopen(filename, 'w');
headers = {'ID','x1','y1','z1','x2','y2','z2','x3','y3','z3',...
           'x4','y4','z4','x5','y5','z5','x6','y6','z6','x7','y7','z7',...
           'x8','y8','z8'};
fprintf(fileID,'%s,',headers{1,1:end-1});
fprintf(fileID,'%s\n',headers{1,end});
fprintf(fileID,strcat('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,',...
        '%d,%d,%d,%d,%d,%d,%d,%d,%d\n'),coords');
fclose(fileID);

% make a zipped version too for easy uploading
zipfile = strrep(filename, '.csv', '.zip');
zip(zipfile, filename);

end

