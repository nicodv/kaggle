function santa()
% Packing Santa's Sleigh Kaggle Competition
% author: Nico de Vos


%% Settings
xlen = 1000;
ylen = 1000;
% Calculate benchmark?
benchmark = 1;


%% Import and Prepare Data
presents = load('presents.mat');
presents = presents.presents;

% Save the columns as separate arrays, representing the IDs, widths,
% lengths, heights and orientation of each present
ID = presents(:,1);
width = presents(:,2);
length = presents(:,3);
height = presents(:,4);
volume = width .* length .* height;
% this is the size of the largest side of the packages
max2dsurf = max([width .* length, length .* height, width .* height], [], 2);
% Note: there are 6 possible orientations, set to 1, meaning the original one
orient = ones(size(ID));

nPresents = size(presents, 1);


%% Kaggle's Benchmark
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
    % the 2 extreme coordinates per present
    coords = zeros(nPresents,2,3);

    for i = 1:nPresents
        % Move to the next row if there isn't room
        if xs + width(i) > xlen + 1 % exceeded allowable width
            % increment y to ensure no overlap
            ys = ys + max(length(lastRowIdxs(1:numInRow)));
            xs = 1;
            numInRow = 0;
        end
        % Move to the next layer if there isn't room
        if ys + length(i) > ylen + 1 % exceeded allowable length
            % increment z to ensure no overlap
            zs = zs - max(height(lastLayerIdxs(1:numInLayer)));
            xs = 1;
            ys = 1;
            numInLayer = 0;
        end
        
        % Fill present coordinate matrix
        coords(i,1,:) = [xs ys zs];
        coords(i,2,:) = [xs+width(i)-1 ys+length(i)-1 zs-height(i)+1];

        % Update location info
        xs = xs + width(i);
        numInRow = numInRow+1;
        numInLayer = numInLayer+1;
        lastRowIdxs(numInRow) = ID(i);
        lastLayerIdxs(numInLayer) = ID(i);
    end

    % We started at z = -1 and went downward, need to shift so all z-values >= 1
    zCoords = coords(:,:,3);
    minZ = min(zCoords(:));
    coords(:,:,3) = zCoords - minZ + 1;
    benchmarkScore = evaluate(coords)
end


%% Heuristic Algorithm
% A smart heuristic encourages a strategy that is better in the long term by:
% - minimizing the average height (obvious)
% - minimizing the standard deviation of the height (i.e., the surface on which
%   to place new presents is as flat/smooth as possible)
% - minimizing free present surface (i.e., the presents are placed against each
%   other with as few gaps as possible)
%
% Other ideas:
% - The order is taken into account because the list is already sorted.
% - An ordering of subgroups of present according to their volumes might help
%   in staying away from local optima. Sum of volumes determines how many presents
%   will be considered at once in a group.
% - Somehow only consider placing presents aligned with other presents.

% function for determining the sum of free box surfaces
surface = @(H) sum(reshape(diff(H, 1, 1),1,[])) + ...
    sum(reshape(diff(H, 1, 2),1,[])) + numel(size(H));

orientOrder = zeros(nPresents, 6);

% matrix with current heights per column
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
    minHLW = min([height(i) length(i) width(i)]);
    minHL = min(height(i), length(i));
    minHW = min(height(i), width(i));
    minLW = min(length(i), width(i));
    if height(i) == minHLW
        if length(i) == minLW
            orientOrder(i,:) = [1 2 3 4 5 6];
        else
            orientOrders(i,:) = [1 2 5 6 3 4];
        end
    elseif length(i) == minHLW
        if height(i) == minHW
            orientOrder(i,:) = [3 4 1 2 5 6];
        else
            orientOrders(i,:) = [3 4 5 6 1 2];
        end
    elseif width(i) == minHLW
        if length(i) == minHL
            orientOrder(i,:) = [5 6 3 4 1 2];
        else
            orientOrders(i,:) = [5 6 1 2 3 4];
        end
    end
    
    % we also prefer a present placed against either another present, or on the
    % borders of the sleigh (this reduces complexity, hopefully still smart enough)

end

meanH = mean(hMat(:));
varH = var(hMat(:));
surfaceH = surface(hMat);

end


%% Statistics and Plots
function metric = evaluate(coords)
% Compute evaluation metric that expresses how well Santa's sleigh is
% packed, judged by the overall compactness of the packing and the
% ordering of the presents: 
% metric  = 2 * max(z-coords) + sigma(order)

nPresents = size(coords, 1);

% Ideal order is the original order
idealOrder = [1:nPresents]';

% Determine the max z-coord; this is the max height of the sleigh
maxZ = max(max(coords(:,2,3)));

% Go down the layers from top to bottom, reorder presents
% in numeric order for each layer
maxZCoord = zeros(nPresents,2);
for i = 1:nPresents
    maxZCoord(i,1) = i;
    maxZCoord(i,2) = max(coords(i,:,3));
end
%sort max z-coord for each present
maxzCoordSorted = sortrows(maxZCoord,[-2 1]);
reOrder = maxzCoordSorted(:,1);

% Compare the new order to the ideal order
order = sum(abs(idealOrder - reOrder));

% Finally compute metric
metric = 2 * maxZ + order;

end


function fRate = fillrate(hMat, gMat)
% Calculate the fill rate of the sleigh

% Determine the max coordinates
maxX = size(hMat, 1);
maxY = size(hMat, 2);
maxZ = max(max(hMat));

encapsVol = maxX * maxY * maxZ;

gaps = sum(gMat(:)) + sum(sum(abs(hMat - maxZ)));

if encapsVol > 0
    fRate = (encapsVol - gaps) / encapsVol;
end

end


%% Check Result and Submission
function collisiondetection(coords)

% Sort by max z-coord for each present
minmaxXYZSorted = sortrows(minmaxXYZCoords,-7);

nPresents = size(coords, 1);
% Compare present i against other presents, checking for collision
for i = 1:nPresents
    for j = i+1:nPresents
        % Test for collision on z-axis first, since sorted
        minZi = minmaxXYZSorted(i,1,3); 
        maxZj = minmaxXYZSorted(j,2,3); 
        if minZi > maxZj
            % No overlap on z-axis implies no collision
            break
        end
        
        % Test for collision on x-axis
        minXi = minmaxXYZSorted(i,1,1);        
        minXj = minmaxXYZSorted(j,1,1);
        maxXi = minmaxXYZSorted(i,2,1); 
        maxXj = minmaxXYZSorted(j,2,1);         
        if or(maxXi < minXj, minXi > maxXj)
            % No overlap on x-axis implies no collision
            continue
        end
        
        % Test for collision on y-axis
        minYi = minmaxXYZSorted(i,1,2); 
        minYj = minmaxXYZSorted(j,1,2); 
        maxYi = minmaxXYZSorted(i,2,2); 
        maxYj = minmaxXYZSorted(j,2,2); 
        if or(maxYi < minYj, minYi > maxYj)
            % No overlap on y-axis implies no collision
            continue
        end
        
        % Overlap on x, y, and z axes indicates collision
        fprintf('Collision check FAILED: packages %d and %d collided\n',...
            minmaxXYZSorted(i,1),minmaxXYZSorted(j,1));
        return
    end
end
fprintf('Collision check PASSED');

end


function createsubmission(filename, coords)

    function allCoords = allcorners()
    % transforms array with 2 coordinates to array with all coordinates
    allCoords(:,1,:) = coords(:,1,:);
    allCoords(:,2,:) = [coords(:,1,1) coords(:,1,2) coords(:,2,3)];
    allCoords(:,3,:) = [coords(:,1,1) coords(:,2,2) coords(:,2,3)];
    allCoords(:,4,:) = [coords(:,1,1) coords(:,2,2) coords(:,1,3)];
    allCoords(:,5,:) = [coords(:,2,1) coords(:,2,2) coords(:,1,3)];
    allCoords(:,6,:) = [coords(:,2,1) coords(:,1,2) coords(:,1,3)];
    allCoords(:,7,:) = [coords(:,2,1) coords(:,1,2) coords(:,2,3)];
    allCoords(:,8,:) = coords(:,2,:);
    end

    fullCoords = allcorners(coords);
    fullCoords = reshape(fullCoords,size(coords, 1), 24);
    fullCoords = [[1:size(coords, 1)]'; fullCoords];
    % Use fprintf to write the header, present IDs, and coordinates to a CSV file.
    fileID = fopen(filename, 'w');
    headers = {'ID','x1','y1','z1','x2','y2','z2','x3','y3','z3',...
               'x4','y4','z4','x5','y5','z5','x6','y6','z6','x7','y7','z7',...
               'x8','y8','z8'};
    fprintf(fileID,'%s,',headers{1,1:end-1});
    fprintf(fileID,'%s\n',headers{1,end});
    fprintf(fileID,strcat('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,',...
            '%d,%d,%d,%d,%d,%d,%d,%d,%d\n'),coords8');
    fclose(fileID);

    % make a zipped version too for easy uploading
    zipfile = strrep(filename, '.csv', '.zip');
    zip(zipfile, filename);

end
