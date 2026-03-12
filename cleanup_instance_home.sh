#!/usr/bin/env bash
# Remove redundant ~/GRiT and ~/detectron2 on the GCP instance.
# Everything needed lives inside citv/ (citv/GRiT and citv/GRiT/third_party/CenterNet2).
# Run from anywhere (uses $HOME). Usage: ./cleanup_instance_home.sh
set -e

echo "This will remove the following from your home directory (if present):"
echo "  $HOME/GRiT"
echo "  $HOME/detectron2"
echo ""
echo "Only citv/ is required; GRiT and detectron2 are already inside citv."
echo "Continue? [y/N]"
read -r r
if [ "$r" != "y" ] && [ "$r" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

freed=0
for dir in GRiT detectron2; do
  if [ -d "$HOME/$dir" ]; then
    echo "Removing $HOME/$dir ..."
    rm -rf "$HOME/$dir"
    echo "  removed."
    freed=1
  else
    echo "  $HOME/$dir not found, skip."
  fi
done

if [ "$freed" = "1" ]; then
  echo "Done. You can run: df -h /  to see freed space."
else
  echo "Nothing to remove."
fi
