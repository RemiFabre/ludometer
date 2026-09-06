#!/usr/bin/env bash
# Turn scripts/game_gif.mjs frames into social clips (gif + mp4), three cuts:
#   table    the whole table, the game accelerated end to end
#   closeup  one board, from the deal to the second scoring
#   zoom     the table, zooming into the player's board at the first scoring
#   scripts/game_gif_render.sh FRAMES_DIR OUT_DIR
set -euo pipefail
F=${1:?frames dir}; O=${2:?out dir}; mkdir -p "$O"
S=$(python3 -c "import json; print(json.load(open('$F/rects.json'))['scale'])")
# rectangles in CSS px from rects.json -> device px
read -r HX HY HW HH MX MY MW MH SY <<<"$(python3 - "$F" <<'PY'
import json,sys; r=json.load(open(sys.argv[1]+"/rects.json")); s=r["scale"]
h=r["human"]; m=r["middle"]; st=r.get("status") or {"y": m["y"]-200}
print(*(int(v*s) for v in (h["x"],h["y"],h["w"],h["h"],m["x"],m["y"],m["w"],m["h"],st["y"])))
PY
)"
FIRST_SCORE=$(python3 -c "
import json; f=json.load(open('$F/frames.json'))
s=[x['i'] for i,x in enumerate(f) if x['scoring'] and not f[i-1]['scoring']] if len(f)>1 else []
print(s[0] if s else 50)")
N=$(ls "$F"/0*.png | wc -l | tr -d ' ')
PAL="split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"

# 1. the whole table (middle + both boards), 16:10, ~8 s
# from the status line ("Your turn: ...") down to the bottom of the boards, 16:10
TW=$((MW + 40*S)); TH=$((TW*10/16)); TX=$((MX - 20*S)); TY=$((HY + HH + 14*S - TH))
ffmpeg -y -loglevel error -framerate 30 -i "$F/%05d.png" \
  -vf "crop=$TW:$TH:$TX:$TY,scale=800:-2,fps=16,$PAL" "$O/table.gif"
ffmpeg -y -loglevel error -framerate 30 -i "$F/%05d.png" \
  -vf "crop=$TW:$TH:$TX:$TY,scale=1280:-2,fps=30" -c:v libx264 -pix_fmt yuv420p -crf 20 "$O/table.mp4"

# 2. close-up on the player's board, deal to the second scoring, ~5 s
CW=$((HW + 24*S)); CH=$((CW*10/16)); CX=$((HX - 12*S)); CY=$((HY + HH/2 - CH/2))
LAST=$(python3 -c "
import json; f=json.load(open('$F/frames.json'))
s=[x['i'] for i,x in enumerate(f) if x['scoring'] and not f[i-1]['scoring']]
print(s[1]+8 if len(s)>1 else len(f)-1)")
ffmpeg -y -loglevel error -framerate 24 -i "$F/%05d.png" -frames:v $LAST \
  -vf "crop=$CW:$CH:$CX:$CY,scale=960:-2,fps=20,$PAL" "$O/closeup.gif"
ffmpeg -y -loglevel error -framerate 24 -i "$F/%05d.png" -frames:v $LAST \
  -vf "crop=$CW:$CH:$CX:$CY,scale=1280:-2,fps=30" -c:v libx264 -pix_fmt yuv420p -crf 20 "$O/closeup.mp4"

# 3. zoom: start on the table, glide into the player's board around the first
#    scoring, glide back out; zoompan on the table crop, z from 1 to ZMAX
ZMAX=$(python3 -c "print(round($TW/$CW, 3))")
A=$((FIRST_SCORE - 18)); B=$((FIRST_SCORE - 4)); C=$((FIRST_SCORE + 22)); D=$((FIRST_SCORE + 36))
ZEXPR="if(lt(in,$A),1,if(lt(in,$B),1+($ZMAX-1)*(in-$A)/($B-$A),if(lt(in,$C),$ZMAX,if(lt(in,$D),$ZMAX-($ZMAX-1)*(in-$C)/($D-$C),1))))"
# focus point: the player's board centre, expressed inside the table crop
FXC=$((HX + HW/2 - TX)); FYC=$((HY + HH/2 - TY))
ffmpeg -y -loglevel error -framerate 30 -i "$F/%05d.png" \
  -vf "crop=$TW:$TH:$TX:$TY,zoompan=z='$ZEXPR':x='$FXC-(iw/zoom)/2':y='$FYC-(ih/zoom)/2':d=1:s=1280x800:fps=30,scale=800:-2,fps=16,$PAL" "$O/zoom.gif"
ffmpeg -y -loglevel error -framerate 30 -i "$F/%05d.png" \
  -vf "crop=$TW:$TH:$TX:$TY,zoompan=z='$ZEXPR':x='$FXC-(iw/zoom)/2':y='$FYC-(ih/zoom)/2':d=1:s=1280x800:fps=30" -c:v libx264 -pix_fmt yuv420p -crf 20 "$O/zoom.mp4"
ls -la "$O" | awk 'NR>1{printf "%7.1f MB  %s\n", $5/1e6, $9}'
