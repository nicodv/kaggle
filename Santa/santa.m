%% Packing Santa's Sleigh Kaggle Competition
% author: Nico de Vos
% 
% Initial analysis shows:
% X, Y and Z sizes range from 2 to 250, mean is 50, std. dev. is 62
% smallest box is 2x2x2, largest 247x249x250
% For reference: 20x20x2500 perfectly ordered boxes
% of size 50x50x50 would score 125,000.

%% Settings
width = 1000;
length = 1000;
% Calculate benchmark?
benchmark = 1;

%% Importing Data
presents = load('presents.mat');

% Save the columns as separate arrays, representing the IDs, widths,
% lengths, heights and orientation of each present
presentID = presents(:,1);
presentWidth = presents(:,2);
presentLength = presents(:,3);
presentHeight = presents(:,4);
% Note: there are 6 possible orientations, set to 1, meaning the original one
% 1 = WLH, Width * Length * Height
% 2 = WHL
% 3 = HLW
% 4 = LWH
% 5 = HWL
% 6 = LHW
presentOrient = ones(size(presentID));

numPresents = size(presents, 1);

%% Initialization

if benchmark = 1;
    %% A Naive Approach
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
    % PresentID and 8 sets of coordinates per present
    presentCoords = zeros(numPresents,25);

    for i = 1:numPresents
        % Move to the next row if there isn't room
        if xs + presentWidth(i) > width + 1 % exceeded allowable width
            % increment y to ensure no overlap
            ys = ys + max(presentLength(lastRowIdxs(1:numInRow)));
            xs = 1;
            numInRow = 0;
        end
        % Move to the next layer if there isn't room
        if ys + presentLength(i) > length + 1 % exceeded allowable length
            % increment z to ensure no overlap
            zs = zs - max(presentHeight(lastLayerIdxs(1:numInLayer)));
            xs = 1;
            ys = 1;
            numInLayer = 0;
        end
        
        % Fill present coordinate matrix
        presentCoords(i,1) = presentIDs(i);
        presentCoords(i,[2 8 14 20]) = xs;
        presentCoords(i,[5 11 17 23]) = xs + presentWidth(i) - 1;
        presentCoords(i,[3 6 15 18]) = ys;
        presentCoords(i,[9 12 21 24]) = ys + presentLength(i) - 1;
        presentCoords(i,[4 7 10 13]) = zs;
        presentCoords(i,[16 19 22 25]) = zs - presentHeight(i) + 1;

        % Update location info
        xs = xs + presentWidth(i);
        numInRow = numInRow+1;
        numInLayer = numInLayer+1;
        lastRowIdxs(numInRow) = presentIDs(i);
        lastLayerIdxs(numInLayer) = presentIDs(i);
    end

    % We started at z = -1 and went downward, need to shift so all z-values >= 1
    zCoords = presentCoords(:,4:3:end);
    minZ = min(zCoords(:));
    presentCoords(:,4:3:end) = zCoords - minZ + 1;
    benchmarkScore = evaluate(presentCoords);


% What follows is a smart initialization procedure that uses the correct order
% of presents, and encourages a strategy that is better in the long term by:
% - minimizing the average height (obvious)
% - minimizing the standard deviation of the height (i.e., the surface on which
%   to place new presents is as flat/smooth as possible)
% - minimizing free present surface (i.e., the presents are placed against each
%   other with as few gaps as possible)
%
% Initialization is done only once, so we're going to approach it a bit brute-force.

% matrix with heights per column
hMat = zeros(width, length);

% matrix with total gaps per column
gMat = zeros(width, length);

%% Statistics and plots initialization


% function for determining the sum of free box surfaces
surface = @(H) sum(diff(H, 1, 1)(:)) + sum(diff(H, 1, 2)(:)) + prod(size(H));
meanH = mean(hMat(:));
varH = var(hMat(:));
surfaceH = surface(hMat);


function metric = evaluate(coords)
% Compute evaluation metric that expresses how well Santa's sleigh is
% packed, judged by the overall compactness of the packing and the
% ordering of the presents: 
% metric  = 2 * max(z-coordinates) + sigma(order)

% Ideal order is the original order
idealOrder = 1:1e6;

numPresents = size(coords, 1);

% Determine the max z-coordinate; this is the max height of the sleigh
maxZ = max(max(coords(:,4:3:end)));

% Go down the layers from top to bottom, reorder presents
% in numeric order for each layer
maxZCoord = zeros(numPresents,2);
for i = 1:numPresents
    maxZCoord(i,1) = coords(i);
    maxZCoord(i,2) = max(coords(i,4:3:end));
end
%sort max z-coord for each present
maxzCoordSorted = sortrows(maxZCoord,[-2 1]);
reOrder = maxzCoordSorted(:,1);

% Compare the new order to the ideal order
order = sum(abs(idealOrder - reOrder));

% Finally compute metric
metric = 2*maxZ + order;


function fillrate(coords)
% Calculate the fill rate of the sleigh
% Determine the max coordinates
maxX = max(max(coords(:,2:3:end-2)));
maxY = max(max(coords(:,3:3:end-1)));
maxZ = max(max(coords(:,4:3:end)));


function createSubmission(filename, presentCoords)
% Use fprintf to write the header, present IDs, and coordinates to a CSV file.
fileID = fopen(filename, 'w');
headers = {'PresentId','x1','y1','z1','x2','y2','z2','x3','y3','z3',...
           'x4','y4','z4','x5','y5','z5','x6','y6','z6','x7','y7','z7',...
           'x8','y8','z8'};
fprintf(fileID,'%s,',headers{1,1:end-1});
fprintf(fileID,'%s\n',headers{1,end});
fprintf(fileID,strcat('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,'...
        '%d,%d,%d,%d,%d,%d,%d,%d,%d\n'),presentCoords');
fclose(fileID);

% make a zipped version too for easy uploading
zipfile = strrep(filename, '.csv', '.zip');
zip(zipfile, filename);
