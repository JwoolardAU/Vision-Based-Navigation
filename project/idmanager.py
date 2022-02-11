import math
from random import randint
import numpy as np
import cv2

class IDManager:
    
    class ID:
    
        def __init__(self, position, IDnum, color):
            """ An ID is definied by its x, y position"""
            self.IDnum = IDnum
            (self.x, self.y) = position
            self.position_history = []
            self.color = color
        
        def update_position(self, position):
            self.position_history.append(position)
            (self.x, self.y) = position
        
        def get_position(self):
            return (self.x,self.y)
        
        def draw_history(self, num_points, img):
            """
                Draw the most recent "num_points" number of points. 
            """
            total_length = len(self.position_history)

            # If the number of points we want to display is 
            # larger than the what is in the list, update num_points to be everything in the list.
            if num_points > total_length:
                num_points = total_length

            max_index = total_length - 1
            stop_index = total_length - num_points

            points = self.position_history

            # Draw "num_points" number of points. 
            # Index of a list syntax: list_name[start:stop:step]
            # Step of -1 goes backwards. 
            for point in points[max_index:stop_index:-1]:
                (x,y) = point
                img = cv2.circle(img, (int(x), int(y)), 4, self.color, -1)
            
            return img

        def draw_total_history(self, img):
            num_items = len(self.position_history)
            updatedImg = self.draw_history(num_items, img)
            return updatedImg
    
    ## End of ID Class##
    ## Start of ID Manager Class ##

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
                distances.append(math.sqrt((x_final - id.x)**2 + (y_final - id.y)**2))
            
            # Take the lowest distance computed and use that as the new ID position
            if distances != []:
                lowest_index = distances.index(min(distances)) 
                updated_position = possiblePositions[lowest_index]
            
                # Remove the position this ID claims, to lessen computations done in future rounds
                del possiblePositions[lowest_index]

                id.update_position(updated_position)

    def draw_ids(self, img):
        for id in self.IDS:
            img = cv2.putText(img, str(id.IDnum), (int(id.x), int(id.y)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(75, 0, 130),thickness=1)
        return img
            


if __name__ == '__main__':

    IDS = IDManager()

    num = input('How many drones? ')
    print()

    initial_positions = []
    for x in range(int(num)):
        print(f"Drone Num: {x+1}")
        x = input('Initial X? ')
        y = input('Initial Y? ')
        print()

        IDS.createID((int(x),int(y)))
    
    num = input("How many possible points do you want to give? ")

    possiblePositions = []
    for x in range(int(num)):
        possiblePositions.append((randint(0,640), randint(0,480)))

    print("\nStarting Positions")
    for id in IDS.IDS:
        print(f"Drone {id.IDnum} starts at {id.get_position()}")

    print("\nPossible Possitions")
    print(possiblePositions)

    IDS.updatePositions(possiblePositions)

    print("\nUpdated Possitions")
    for id in IDS.IDS:
        print(f"ID {id.IDnum} is 'now' at {id.get_position()}")
    


    



        

                


                







"""
    Assign an ID to the drone at the beginning. 
    The ID begins with the Initial postion of the drone.

    To Update the ID, pass a list of potential points
    and select the point that is closest to the ID.
    Update the ID's position.  
"""

"""
    Rather than passing the points to each ID, 
    Make a manager that will figure it out..

    Collect a list of the drones positions, 

    To Update, pass the lists of potential points, 

"""
