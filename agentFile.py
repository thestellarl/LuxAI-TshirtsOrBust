from re import S
from sklearn import cluster
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import sys
from sklearn.cluster import DBSCAN
import numpy as np
import logging

logging.FileHandler('agent.log', mode='w')
logging.basicConfig(filename='agent.log', encoding='utf-8', level=logging.DEBUG)

def get_resource_clusters(inputArray):
    clustering = DBSCAN(math.sqrt(8), min_samples=2).fit(inputArray)
    labels = clustering.labels_
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    output = []
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = inputArray[class_member_mask & core_samples_mask]
        output.append(xy)
        # logging.info(f'{k} - {xy}')
    return output
    
def get_best_cluster(clusters, unit):
    MASS_CONSTANT = 2
    DISTANCE_CONSTANT = 2
    if len(clusters) == 0: return None
    ret = clusters[0]
    max_score = 0
    for i, c in enumerate(clusters):
        # logging.info(f'{np.sum(c, axis=0) / len(c) }')
        logging.info(f'{c}')
        cluster_avg = np.sum(c, axis=0) / len(c)
        score = MASS_CONSTANT * len(c) + DISTANCE_CONSTANT / (math.pow(unit.pos.x - cluster_avg[0], 2) + math.pow(unit.pos.y - cluster_avg[1], 2))
        if score > max_score:
            max_score = score
            ret = c
    return ret

def get_cluster_perimeter(cluster: list):
    perimeter = set()
    for c in cluster:
        perimeter.add(tuple(c + [1, 0]))
        perimeter.add(tuple(c + [0, 1]))
        perimeter.add(tuple(c + [-1, 0]))
        perimeter.add(tuple(c + [0, -1]))
    for c in cluster:
        perimeter.remove(tuple(c))
    return np.array([[x, y] for (x,y) in perimeter])

def get_direction(source, destination):
    dist = np.abs(source - destination)
    if dist[0] > dist[1]:
        return 'e' if destination[0] > source[0] else 'w'
    else:
        return 's' if destination[0] > source[0] else 'n'

# we declare this global game_state object so that state persists across turns so we do not need to reinitialize it all the time
game_state = None
unit_controller = {}
def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    wood_tiles = np.array([np.array([cell.pos.x, cell.pos.y]) for cell in resource_tiles if cell.resource.type == Constants.RESOURCE_TYPES.WOOD])
    coal_tiles = np.array([np.array([cell.pos.x, cell.pos.y]) for cell in resource_tiles if cell.resource.type == Constants.RESOURCE_TYPES.COAL])
    uranium_tiles = np.array([np.array([cell.pos.x, cell.pos.y]) for cell in resource_tiles if cell.resource.type == Constants.RESOURCE_TYPES.URANIUM])
    
    wood_clusters = get_resource_clusters(wood_tiles)
    coal_clusters = get_resource_clusters(coal_tiles)
    uranium_clusters = get_resource_clusters(uranium_tiles)

    for i, unit in enumerate(player.units):
        if unit.id not in unit_controller:
            unit.pos.direction_to
            best_cluster = get_best_cluster(wood_clusters, unit)
            # logging.info(f'{get_cluster_perimeter(best_cluster)}')
            unit_controller[unit.id] = {}
            unit_controller[unit.id]['target'] = best_cluster[np.sum(np.square(np.abs(best_cluster - [unit.pos.x, unit.pos.y])), axis=1).argmin()]
            logging.info(f'{unit_controller[unit.id]["target"]}')
            actions.append(unit.move(get_direction([unit.pos.x, unit.pos.y], unit_controller[unit.id]['target'])))
        # logging.info(f'{unit.id} - {unit_controller[unit.id].target}')
    # add debug statements like so!
    if game_state.turn == 0:
        print("Agent is running!", file=sys.stderr)
        actions.append(annotate.circle(0, 0))
    return actions