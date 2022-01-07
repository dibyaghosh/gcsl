read s
for e in pointmass_empty pointmass_rooms lunar pusher door 
do
echo Environment $e Seed $s
python experiments/gcsl_example.py -S $s -E $e &
done

