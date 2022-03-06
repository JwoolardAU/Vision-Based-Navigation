import math
from random import randint
import cv2


class IDManager:
    """
        Funciton differentiate between multiple objects of the same category
    """

    class ID:
        """
            An ID will be defined by its x, y position
        """
        def __init__(self, position, id_num, color):

            self.id_num = id_num
            self.position = position
            (self.x, self.y) = position
            self.color = color

            self.position_history = []
            self.position_history.append(position)

            # Attritubtes used to compute direction vectors
            self.flag_point = (0, 0)
            self.starting_position = (0, 0)
            self.ending_posiiton = (0, 0)
            self.dir_vector = (0, 0)
            self.flag_vector = (0, 0)
            self.turn_angle = 0

        def update_position(self, position):
            """ Update the position history and position attribute"""
            self.position_history.append(position)
            (self.x, self.y) = position
            self.position = position

        def draw_history(self, num_points, img):
            """ Draw the most recent points for this ID
                num_points is how many points to draw
            """

            total_length = len(self.position_history)

            # If the number of points we want to display is
            # larger than the what is in the list,
            # update num_points to be everything in the list.
            if num_points > total_length:
                num_points = total_length

            max_index = total_length - 1
            stop_index = total_length - num_points

            # Index of a list syntax: list_name[start:stop:step]
            # Step of -1 goes backwards.
            for point in self.position_history[max_index:stop_index:-1]:
                (x, y) = point
                img = cv2.circle(img, point, 4, self.color, -1)

            return img

    # End of ID Class ##
    # Start of ID Manager Class ##

    def __init__(self):
        """
            The manager will keep track of all the active ID's
            by using a list of active IDs.
        """
        self.IDS = []
        self.IDnum = 0

    def createID(self, position, color):
        """
            When given a new position, increment the IDnum
            and assign the new ID a position.
        """
        self.IDnum += 1
        self.IDS.append(self.ID(position, self.IDnum, color))

    def updatePositions(self, possiblePositions):
        """
        Compute the distances from each ID to all the possible
        ID points. The lowest distance one is probably the same ID,
        so update the ID's position to the smallest delta_d.
        """
        # For every ID we are tracking
        for id in self.IDS:
            distances = []
            # Look at every possible point the ID could have moved to
            for possiblePosition in possiblePositions:
                # Compute the distance from every point and add to a list
                (x_final, y_final) = possiblePosition
                distances.append(math.sqrt((x_final - id.x)**2 +
                                           (y_final - id.y)**2))

            # Take the lowest distance computed
            # and use that as the new ID position
            if distances != []:
                lowest_index = distances.index(min(distances))
                updated_position = possiblePositions[lowest_index]

                # Remove the position this ID claims,
                # to lessen computations done in future rounds
                del possiblePositions[lowest_index]

                id.update_position(updated_position)

    def updateFlags(self, flags):
        if flags:
            for id in self.IDS:
                id.flag_point = flags[0]

    def draw_id_nums(self, img):
        for id in self.IDS:
            img = cv2.putText(img, str(id.id_num), id.position,
                              fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                              fontScale=1, color=id.color, thickness=1)
        return img

    def get_current_positions(self):
        """
            Function to get the position of every ID
        """
        positions = []
        for id in self.IDS:
            positions.append(id.position_history[-1])

        return positions

    def set_starting_positions(self):

        for id in self.IDS:
            id.starting_position = id.position_history[-1]
        return None

    def set_ending_positions(self):

        for id in self.IDS:
            id.ending_position = id.position_history[-1]
        return None

    def assign_destination(self, destinations):
        """
            Function to assign the ID's a number of destinations
            based on the total num of destinations.

            Idea:
                Get drone points, and determine what "flag"
                each drone is closesst to.
                Assign the remaining flags to the two drones by which
                flag is closest to the flag that the drone is closest to.
        """
        pass

    def draw_vectors(self, img):
        output = img
        for id in self.IDS:
            # Pre moving and Flag positions
            output = cv2.circle(output, id.starting_position,
                                5, (255, 0, 0), -1)
            output = cv2.circle(output, id.flag_point,
                                5, (0, 255, 0), -1)
            # Vectors
            dirPoint2 = (id.ending_position[0] + id.dir_vector[0],
                         id.ending_position[1] + id.dir_vector[1])
            output = cv2.line(output, id.ending_position,
                              dirPoint2, (255, 0, 255), 3)
            flagPoint2 = (id.ending_position[0] + id.flag_vector[0],
                          id.ending_position[1] + id.flag_vector[1])
            output = cv2.line(output, id.ending_position,
                              flagPoint2, (0, 0, 255), 3)

        return output

    def draw_IDS_history(self, img, num_points):

        for id in self.IDS:
            img = id.draw_history(20, img)

        return img


def example():
    IDS = IDManager()

    num = input('How many drones? ')
    print()

    for x in range(int(num)):
        print(f"Drone Num: {x+1}")
        x = input('Initial X? ')
        y = input('Initial Y? ')
        print()

        IDS.createID((int(x), int(y)))

    num = input("How many possible points do you want to give? ")

    possiblePositions = []
    for x in range(int(num)):
        possiblePositions.append((randint(0, 640), randint(0, 480)))

    print("\nStarting Positions")
    for id in IDS.IDS:
        print(f"Drone {id.IDnum} starts at {id.get_position()}")

    print("\nPossible Possitions")
    print(possiblePositions)

    IDS.updatePositions(possiblePositions)

    print("\nUpdated Possitions")
    for id in IDS.IDS:
        print(f"ID {id.IDnum} is 'now' at {id.get_position()}")


if __name__ == '__main__':

    example()
