using DataFrames
using DataArrays
using Stats
using NumericExtensions

# Import the data and set up data frames
presents = readtable("data/presents.csv")

nPresents = nrow(presents)

ID = presents["PresentId"]
width = presents["Dimension1"]
Length = presents["Dimension2"]
height = presents["Dimension3"]

volume = width .* length .* height
minVol = minimum(volume)
maxVol = maximum(volume)

# Solution
sleighWidth = 1000
sleighLength = 1000

xs = 1
ys = 1
zs = 1

lastRowIdxs = fill!([1:1000],0)
lastLayerIdxs = fill!([1:1000],0)

numInRow = 0
numInLayer = 0

coords = DataFrame(PresentId = fill!([1:nPresents],0), x1 = fill!([1:nPresents],0), y1 = fill!([1:nPresents],0), z1 = fill!([1:nPresents],0), x2 = fill!([1:nPresents],0), y2 = fill!([1:nPresents],0), z2 = fill!([1:nPresents],0), x3 = fill!([1:nPresents],0), y3 = fill!([1:nPresents],0), z3 = fill!([1:nPresents],0), x4 = fill!([1:nPresents],0), y4 = fill!([1:nPresents],0), z4 = fill!([1:nPresents],0), x5 = fill!([1:nPresents],0), y5 = fill!([1:nPresents],0), z5 = fill!([1:nPresents],0), x6 = fill!([1:nPresents],0), y6 = fill!([1:nPresents],0), z6 = fill!([1:nPresents],0), x7 = fill!([1:nPresents],0), y7 = fill!([1:nPresents],0), z7 = fill!([1:nPresents],0), x8 = fill!([1:nPresents],0), y8 = fill!([1:nPresents],0), z8 = fill!([1:nPresents],0))

for (i in 1:nPresents)

    # Move to the next row if there isn't room
    if (xs + width[i] > sleighWidth + 1)
        # exceeded allowable width
        ys = ys + maximum(length[lastRowIdxs[1:numInRow]]) # increment y to ensure no overlap
        xs = 1
        numInRow = 1
    end

    # Move to the next layer if there isn't room
    if (ys + length[i] > sleighLength + 1)
        # exceeded allowable length
        zs = zs - maximum(height[lastLayerIdxs[1:numInLayer]]) # increment z to ensure no overlap
        xs = 1
        ys = 1
        numInLayer = 0
    end

    # Fill present coordinate matrix
    coords[i,1] = presentIDs[i]
    coords[i,[2,8,14,20]] = xs
    coords[i,[5,11,17,23]] = xs + width[i] - 1
    coords[i,[3,6,15,18]] = ys
    coords[i,[9,12,21,24]] = ys + length[i] - 1
    coords[i,[4,7,10,13]] = zs
    coords[i,[16,19,22,25]] = zs - height[i] + 1

    # Update location info
    xs = xs + width[i]
    numInRow = numInRow + 1
    numInLayer = numInLayer + 1
    lastRowIdxs[numInRow] = presentIDs[i]
    lastLayerIdxs[numInLayer] = presentIDs[i]

end

# We started at z = -1 and went downward, need to shift so all z-values >= 1
zCoords = coords[1:end,[4,7,10,13,16,19,22,25]]
minZ = minimum([coords["z1"],coords["z2"],coords["z3"],coords["z4"],coords["z5"],coords["z6"],coords["z7"],coords["z8"]])
coords[1:end,[4,7,10,13,16,19,22,25]] = zCoords - minZ + 1

# Evaluation metric
function getScore(submission)

submission_Z = submission[1:end,[1,4,7,10,13,16,19,22,25]]
submission_Z["maxZ"] = max(submission["z1"],submission["z2"],submission["z3"],submission["z4"],submission["z5"],submission["z6"],submission["z7"],submission["z8"])
sort!(submission_Z, cols=(order("maxZ", rev=true),"PresentId"))
submission_Z["presentOrder"] = 1:nrow(submission_Z)

println("Evaluation metric score is: ", 2 * maximum(submission_Z["maxZ"] + sum(abs(submission_Z["PresentId"] - submission_Z["presentOrder"]))))

end

# Output score and save solution
getScore(coords)
writetable("data/output.csv", coords)
println("Solution saved to data/output.csv")
