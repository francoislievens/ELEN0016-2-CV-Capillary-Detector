from cytomine import Cytomine
from cytomine.models import *
from cytomine.models.image import SliceInstanceCollection
from shapely import wkt
from shapely.affinity import affine_transform
from tqdm import tqdm
import config
import os
import sys

VERBOSE = False



def download_annotations(public_key, private_key, data_path):

    host = "https://learn.cytomine.be"
    conn = Cytomine.connect(host, public_key, private_key)

    # ... Connect to Cytomine (same as previously) ...
    projects = ProjectCollection().fetch()


    for project in projects:

        print('### GROUP: {} ###'.format(project.name))
        name = project.name

        # Check if folder already exists:
        if os.path.exists('{}/annotations/{}'.format(data_path, name)):
            # Delete it
            try:
                os.remove('{}/annotations/{}/{}.csv'.format(data_path, name, name))
            except:
                print('File {}.csv already deleted'.format(name))
            os.rmdir('{}/annotations/{}'.format(data_path, name))

        # Create directory, file and add headers
        os.mkdir('{}/annotations/{}'.format(data_path, name))
        output = open('{}/annotations/{}/{}.csv'.format(data_path, name, name), 'a')
        output.write('ID;Image;Project;Terms;Track;Slice;Geometry\n')

        images = ImageInstanceCollection().fetch_with_filter("project", project.id)

        terms = TermCollection().fetch_with_filter("project", project.id)
        slices = SliceInstanceCollection().fetch_with_filter("imageinstance", images[0].id)

        annotations = AnnotationCollection()
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showTerm = True
        annotations.showTrack = True
        annotations.showImage = True
        annotations.showSlice = True
        annotations.project = project.id

        annotations.fetch()

        for annot in annotations:

            if VERBOSE:
                print("ID: {} | Image: {} | Project: {} | Terms: {} | Track: {} | Slice: {}".format(
                    annot.id, annot.image, annot.project, terms.find_by_attribute("id", annot.term[0]), annot.track, annot.time))

            # Write in file:
            output.write("{};{};{};{};{};{};".format(
                annot.id, annot.image, annot.project, terms.find_by_attribute("id", annot.term[0]), annot.track, annot.time))

            geometry = wkt.loads(annot.location)
            # print("Geometry from Shapely (cartesian coordinate system): {}".format(geometry))

            # In OpenCV, the y-axis is reversed for points.
            # x' = ax + by + x_off => x' = x
            # y' = dx + ey + y_off => y' = -y + image.height
            # matrix = [a, b, d, e, x_off, y_off]
            image = images.find_by_attribute("id", annot.image)
            geometry_opencv = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])
            #print(geometry_opencv)

            # Write geometry in the file
            output.write('{}\n'.format(geometry_opencv))

                # print("Geometry with OpenCV coordinate system: {}".format(geometry_opencv))

        output.close()

def download_images(public_key, private_key, data_path):

    host = "https://learn.cytomine.be"
    conn = Cytomine.connect(host, public_key, private_key)

    projects = ProjectCollection().fetch()
    for project in projects:
        image_instances = ImageInstanceCollection().fetch_with_filter("project", project.id)
        path = os.path.join('{}/images'.format(data_path), project.name, image_instances[0].originalFilename)
        image_instances[0].download(path, override=False)

def prepare_data_folders(data_path):

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists('{}/annotations'.format(data_path)):
        os.mkdir('{}/annotations'.format(data_path))
    if not os.path.exists('{}/images'.format(data_path)):
        os.mkdir('{}/images'.format(data_path))

if __name__ == '__main__':

    public_key = '5b827fdf-3d3a-4cfc-a831-8016b0ec122d'
    private_key = 'b52b59a9-95e4-4d95-b968-951b82860b50'

    data_path = config.DATA_PATH
    download_annotations(public_key, private_key, data_path)

    download_images(public_key, private_key, data_path)