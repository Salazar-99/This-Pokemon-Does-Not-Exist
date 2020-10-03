#Create jpg from each png (requires imagemagick)
mogrify -format jpg *.png
#Remove all png files
rm *.png