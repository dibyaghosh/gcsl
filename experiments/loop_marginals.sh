
for e in pointmass_empty pointmass_rooms lunar pusher door
do 
    for s in 0 1 2
    do
	echo Environment $e Seed $s
        python experiments/gcsl_example_n1.py -S $s -E $e &
    done
done
