kill -9 $(ps -ef|grep detection|awk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
