import concurrent.futures
import cv2
from model import Model


def main():
    """ Entry to the program"""

    PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
    PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

    # How many boxes do we expect?
    MAX_BOXES = 6
    # How confident does the model need to be to display any bouding box?
    MIN_SCORE_THRESH = .70

    cam = cv2.VideoCapture(0)

    Tf_Model = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS,
                     cam, MAX_BOXES, MIN_SCORE_THRESH)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(Tf_Model.update_vectors_thread)

        while True:
            cv2.imshow("Vector", Tf_Model.bounding_box_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean Exit
        cv2.destroyAllWindows()
        Tf_Model.cam.release()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
