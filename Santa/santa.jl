# Packing Santa's Sleigh, Kaggle competition
# author: Nico de Vos

using DataFrames
using NumericExtensions
using Stats


##### Settings #####
sleighWidth = 1000
sleighLength = 1000
# Calculate benchmarks?
benchmark = 1


##### Import and prepare data #####
# read into DataFrame
presents = readtable("presents.csv")

nPresents = nrow(presents)

ID = presents["PresentId"]
width = presents["Dimension1"]
length = presents["Dimension2"]
height = presents["Dimension3"]

volume = dot(width, length, height)
# the size of the largest side of the presents
max2dsurf = maximum([dot(width, length); dot(length, height); dot(width, height)], 1)
# Note: there are 6 possible orientations, initialize to 1, meaning the original one
orient = fill(1, nPresents)


##### Benchmark #####
if benchmark = 1
    # Initial coordinates for placing presents
    xs, ys = 1
    zs = -1

    lastRowIdxs = zeros(UInt, 500)
    lastLayerIdxs = zeros(UInt, 500)
    numInRow = 0
    numInLayer = 0

    # the 2 extreme coordinates per present
    coords = zeros(UInt, (nPresents, 2, 3))

    for i in 1:nPresents

        # Move to the next row if there isn't room
        if (xs + width[i] > sleighWidth + 1)
            # exceeded allowable width
            # increment y to ensure no overlap
            ys = ys + maximum(length[lastRowIdxs[1:numInRow]])
            xs = 1
            numInRow = 0
        end

        # Move to the next layer if there isn't room
        if (ys + length[i] > sleighLength + 1)
            # exceeded allowable length
            # increment z to ensure no overlap
            zs = zs - maximum(height[lastLayerIdxs[1:numInLayer]])
            xs, ys = 1
            numInLayer = 0
        end

        # Fill present coordinate matrix
        coords[i,1,:] = [xs ys zs]
        coords[i,2,:] = [xs+width[i]-1 ys+length[i]-1 zs-height[i]+1]

        # Update location info
        xs = xs + width[i]
        numInRow = numInRow + 1
        numInLayer = numInLayer + 1
        lastRowIdxs[numInRow] = ID[i]
        lastLayerIdxs[numInLayer] = ID[i]

    end

    # We started at z = -1 and went downward, need to shift so all z-values >= 1
    zCoords = coords[:,:,3]
    minZ = minimum(zCoords)
    coords[:,:,3] = zCoords - minZ + 1

    benchmarkScore = calcscore(coords)
    println("Benchmark score: $benchmarkScore")

end


##### Heuristic algorithm #####
# A smart heuristic encourages a strategy that is better in the long term by:
# - minimizing the average height (obvious)
# - minimizing the standard deviation of the height (i.e., the surface on which
# to place new presents is as flat/smooth as possible)
# - minimizing free present surface (i.e., the presents are placed against each
# other with as few gaps as possible)
#
# Other ideas:
# - The order is taken into account because the array is already sorted in order.
# - An ordering of subgroups of present according to their volumes might help
#   in staying away from local optima. Sum of largest surfaces determines how many
#   presents will be considered at once in a group.
# - Somehow only consider placing presents aligned with other presents.

# matrix with current heights per column
H = zeros(UInt, (sleighWidth, sleighLength))

# matrix with total gaps per column
G = zeros(UInt, (sleighWidth, sleighLength))

orientOrder = zeros(UInt, (nPresents, 6));

for i = 1:nPresents
    # determine preferred order of orientation: we prefer to place
    # presents as flat as possible (no preference in x/y dimensions)
    # 1 = WLH, width * length * height
    # 2 = LWH
    # 3 = WHL
    # 4 = HWL
    # 5 = LHW
    # 6 = HLW
    minHLW = min([height(i) length(i) width(i)])
    minHL = min(height(i), length(i))
    minHW = min(height(i), width(i))
    minLW = min(length(i), width(i))
    if height(i) = minHLW
        if length(i) = minLW
            orientOrder[i,:] = [1 2 3 4 5 6]
        else
            orientOrder[i,:] = [1 2 5 6 3 4]
        end
    elseif length(i) = minHLW
        if height(i) = minHW
            orientOrder[i,:] = [3 4 1 2 5 6]
        else
            orientOrder[i,:] = [3 4 5 6 1 2]
        end
    elseif width(i) = minHLW
        if length(i) = minHL
            orientOrder[i,:] = [5 6 3 4 1 2]
        else
            orientOrder[i,:] = [5 6 1 2 3 4]
        end
    end
    
    # we also prefer a present placed against either another present, or on the
    # borders of the sleigh (this reduces complexity, hopefully still smart enough)

end


##### Statistics #####
function objective(H, G)

    # function for determining the sum of free box surfaces
    # note: gaps are overweighted since they each count for 8 surfaces
    # even when multiple gaps are connected
    surface(hMat, gMat) = sum(abs(diff(hMat, 1))) + sum(abs(diff(hMat, 2))) + length(hMat) + 8 * sum(gMat)

    meanH = abs(mean(H))
    varH = var(H)
    surfaceH = surface(H, G)

    # this is the aggregate objective function that weighs the individual aspects
    objFun = meanH ^ 2 + varH + surfaceH

end


function calcscore(coords)
    # Calculate score on competition metric

    # Ideal order is the original order
    idealOrder = [1:nPresents]

    # Determine the max z-coord; this is the max height of the sleigh
    maxZ = maximum(coords[:,2,3])

    # Go down the layers from top to bottom, reorder presents
    # in numeric order for each layer
    maxZCoord = zeros(Int, [nPresents, 2])
    for i = 1:nPresents
        maxZCoord(i,1) = i
        maxZCoord(i,2) = maximum(coords[i,:,3])
    end

    # sort max z-coord for each present
    maxzCoordSorted = sort(maxZCoord,[-2 1])
    reOrder = maxzCoordSorted(:,1)

    # Compare the new order to the ideal order
    order = sum(abs(idealOrder - reOrder))

    # Finally compute metric
    metric = 2*maxZ + order

    println("Evaluation metric score is: $metric")

end


function fillrate(H, G)
    # Calculate the fill rate of the sleigh

    # calculate volume of smallest encapsulating box
    maxZ = max(H)
    encapsVol = sleighWidth * sleighLength * maxZ

    # counts all gaps, including empty spaces below the maximum height
    gaps = sum(G) + sum(abs(H - maxZ))

    if (encapsVol > 0)
        return (encapsVol - gaps) / encapsVol;
    end

end


##### Generate submission #####
score = calcscore(coords)
println("Solution scored $score")
writetable("submission.csv", coords)
println("Solution saved")
