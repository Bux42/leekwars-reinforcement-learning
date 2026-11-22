## Main long term goal

Having a leekwars fight reinforcement learning environment capable of training any kind of agent (different equipments, stats, weapons and chips) in any kind of fight (1v1, 4v4, 6v6, battle royal etc) in Leekwars.

The agent will play each playing entity, it will control the summons, plants, and will use self play to continuously improve.

The fight executor (at the moment, it is the leekwars-generator-c submodule) is focused on performances, with cloning features for a future potential monte carlo tree search implementation.
The reason behind a low level implementation is that training will take place on consumer hardware, on the high end of gaming PCs, and as it might take millions of fights to train a model, we must make sure that we're efficient.
The fight executor should have 100% parity with the official game, else the results would lose a lot of meaning, as we would be training on a "different" game.
