for d in `ls att_faces`;
do 
for f in `ls ./att_faces/$d`;
do
convert -resize 200x200! ./att_faces/$d/$f ./att_faces/$d/$f;
done;
done;
