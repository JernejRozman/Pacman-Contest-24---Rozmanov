# my_team.py

from capture_agents import CaptureAgent
import random
from game import Directions
import util

###########################
#       Rozmanov        #
###########################

def create_team(indexes, num, is_red,
                first='OfenzivniAgent', second='DefenzivniAgent'):
    """
    Funkcija za ustvarjanje ekipe dveh agentov, ki se bosta uporabljala v igri.

    :param indexes: Indeksi agentov.
    :param num: Število agentov.
    :param is_red: Ali je ekipa rdeča.
    :param first: Ime prvega agenta.
    :param second: Ime drugega agenta.
    :return: Seznam agentov.
    """
    return [eval(first)(indexes[0]), eval(second)(indexes[1])]

class OfenzivniAgent(CaptureAgent):
    """
    Ofenzivni agent, ki uporablja A* algoritem za zbiranje hrane na nasprotnikovi strani.
    """

    def chooseAction(self, gameState):
        """
        Izbira naslednje dejanje agenta na podlagi trenutnega stanja igre.

        :param gameState: Trenutno stanje igre.
        :return: Izbrana akcija.
        """
        # Dobimo trenutni položaj agenta
        position = gameState.getAgentPosition(self.index)

        # Pridobimo seznam hrane na nasprotnikovi strani
        food = self.getFood(gameState).asList()

        # Če je na voljo hrana, poiščemo pot do najbližje hrane z A* algoritmom
        if food:
            cilj = min(food, key=lambda x: self.getMazeDistance(position, x))
            akcija = self.a_star_search(gameState, position, cilj)
            if akcija:
                return akcija

        # Če ni hrane ali ni mogoče najti poti, se premaknemo naključno
        return random.choice(gameState.getLegalActions(self.index))

    def a_star_search(self, gameState, start, goal):
        """
        Izvede A* iskanje od začetne pozicije do cilja.

        :param gameState: Trenutno stanje igre.
        :param start: Začetna pozicija.
        :param goal: Ciljna pozicija.
        :return: Prva akcija na poti do cilja.
        """
        # Definiramo funkcijo heuristike (Manhattan razdalja)
        def heuristic(pos, goal):
            return self.getMazeDistance(pos, goal)

        # Ustvarimo prioritetno vrsto
        frontier = util.PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()

        while not frontier.isEmpty():
            current_position, actions = frontier.pop()

            if current_position in explored:
                continue

            explored.add(current_position)

            if current_position == goal:
                if actions:
                    return actions[0]  # Vrni prvo akcijo na poti
                else:
                    return Directions.STOP

            for action in gameState.getLegalActions(self.index):
                if action == Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(self.index, action)
                successor_position = successor.getAgentPosition(self.index)
                if successor_position in explored:
                    continue
                cost = len(actions + [action]) + heuristic(successor_position, goal)
                frontier.push((successor_position, actions + [action]), cost)

        return None

class DefenzivniAgent(CaptureAgent):
    """
    Defenzivni agent, ki uporablja A* algoritem za sledenje sovražnim agentom.
    """

    def chooseAction(self, gameState):
        """
        Izbira naslednje dejanje agenta na podlagi trenutnega stanja igre.

        :param gameState: Trenutno stanje igre.
        :return: Izbrana akcija.
        """
        # Dobimo položaje sovražnih agentov
        sovrazniki = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        napadalci = [a for a in sovrazniki if a.isPacman and a.getPosition() != None]

        # Če so napadalci na naši strani, jih poskušamo ujeti
        if napadalci:
            pozicije = [a.getPosition() for a in napadalci]
            najblizji_napadalec = min(pozicije, key=lambda x: self.getMazeDistance(gameState.getAgentPosition(self.index), x))
            akcija = self.a_star_search(gameState, gameState.getAgentPosition(self.index), najblizji_napadalec)
            if akcija:
                return akcija

        # Sicer patruliramo po svoji polovici
        else:
            # Določimo točko patrulje (sredina naše strani)
            patruljne_tocke = self.get_patrol_points(gameState)
            cilj = random.choice(patruljne_tocke)
            akcija = self.a_star_search(gameState, gameState.getAgentPosition(self.index), cilj)
            if akcija:
                return akcija

        # Če vse drugo odpove, se premaknemo naključno
        return random.choice(gameState.getLegalActions(self.index))

    def a_star_search(self, gameState, start, goal):
        """
        Izvede A* iskanje od začetne pozicije do cilja.

        :param gameState: Trenutno stanje igre.
        :param start: Začetna pozicija.
        :param goal: Ciljna pozicija.
        :return: Prva akcija na poti do cilja.
        """
        # Definiramo funkcijo heuristike (Manhattan razdalja)
        def heuristic(pos, goal):
            return self.getMazeDistance(pos, goal)

        # Ustvarimo prioritetno vrsto
        frontier = util.PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()

        while not frontier.isEmpty():
            current_position, actions = frontier.pop()

            if current_position in explored:
                continue

            explored.add(current_position)

            if current_position == goal:
                if actions:
                    return actions[0]  # Vrni prvo akcijo na poti
                else:
                    return Directions.STOP

            for action in gameState.getLegalActions(self.index):
                if action == Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(self.index, action)
                successor_position = successor.getAgentPosition(self.index)
                if successor_position in explored:
                    continue
                cost = len(actions + [action]) + heuristic(successor_position, goal)
                frontier.push((successor_position, actions + [action]), cost)

        return None

    def get_patrol_points(self, gameState):
        """
        Določi točke na naši strani, ki jih bo agent patruliral.

        :param gameState: Trenutno stanje igre.
        :return: Seznam patruljnih točk.
        """
        width = gameState.getWalls().width
        height = gameState.getWalls().height
        mid_x = width // 2 - 1 if self.red else width // 2

        valid_positions = []
        for y in range(1, height - 1):
            if not gameState.hasWall(mid_x, y):
                valid_positions.append((mid_x, y))

        return valid_positions
