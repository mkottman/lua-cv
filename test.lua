local cv = require 'luacv'

cap = cv.CaptureFromCAM()
print(cap, cap.width, cap.height, cap.fps)

local chessSize = {8, 6}

cv.NamedWindow 'vid'
cv.NamedWindow 'dbg'

local key
while key ~= 27 do
	key = cv.WaitKey(100)
	local img = cap:QueryFrame()
	if img then
		if key == 32 then
			local corners, count = cv.FindChessboardCorners(img, chessSize)
			print('Found corners:', count)
			if corners and count > 0 then
				local size = cv.GetSize(img)
				print('Size:', size[1], size[2])
				local mono = cv.CreateImage(size, '8u', 1)
				cv.CvtColor(img, mono, 'rgb2gray')
				cv.FindCornerSubPix(mono, corners, 4, {iter=10, eps=0.5})
				cv.DrawChessboardCorners(mono, chessSize, corners)
				cv.ShowImage('dbg', mono)
			end
		end
		cv.ShowImage("vid", img)
	end
end
