for s in 0 1 2
do 
    for e in pointmass_empty pointmass_rooms lunar pusher door 
    do
	echo Environment $e Seed $s
        python experiments/gcsl_example_n11.py -S $s -E $e &
    done
    wait
done

#
#for s in 1
#do 
#    for e in pointmass_empty pointmass_rooms lunar pusher door 
#    do
#	echo Environment $e Seed $s
#        python experiments/gcsl_example_n.py -S $s -E $e &
#    done
#done
#
#for s in 2
#do 
#    for e in pointmass_empty pointmass_rooms lunar pusher door 
#    do
#	echo Environment $e Seed $s
#        python experiments/gcsl_example_n.py -S $s -E $e &
#    done
#done


exit 0
