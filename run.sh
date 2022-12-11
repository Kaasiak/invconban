for agent in 0 1 11 12 2 21 3 
do
    for key in {0..4}
    do
        echo "running main-bicb.py for agent$agent-key$key ..."
        python src/main-bicb.py -i "data/icb-syntdata/agent$agent-key$key.obj" -o "res/bicb-agent$agent-key$key.obj"
        echo "running main-nbicb.py for agent$agent-key$key ..."
        python src/main-nbicb.py -i "data/icb-syntdata/agent$agent-key$key.obj" -o "res/nbicb-agent$agent-key$key.obj"
        
        echo "running main-optimistic-bicb.py for agent$agent-key$key ..."
        python src/main-optimistic-bicb.py -i "data/icb-syntdata/agent$agent-key$key.obj" -o "res/optimistic/bicb-agent$agent-key$key.obj"

        echo "running main-greedy-bicb.py for agent$agent-key$key ..."
        python src/main-greedy-bicb.py -i "data/icb-syntdata/agent$agent-key$key.obj" -o "res/greedy/bicb-agent$agent-key$key.obj"
    done
done
