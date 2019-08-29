#!/bin/bash

# airplane automobile bird cat deer dog frog horse ship truck

: '
cd automobile
mv *.png ../
cd ..
rm -r automobile/

cd airplane
mv *.png ../
cd ..
rm -r airplane/

cd bird
mv *.png ../
cd ..
rm -r bird/

cd cat
mv *.png ../
cd ..
rm -r cat/

cd deer
mv *.png ../
cd ..
rm -r deer/

cd dog
mv *.png ../
cd ..
rm -r dog/

cd frog
mv *.png ../
cd ..
rm -r frog/

cd horse
mv *.png ../
cd ..
rm -r horse/

cd ship
mv *.png ../
cd ..
rm -r ship

cd truck
mv *.png ../
cd ..
rm -r truck
'
