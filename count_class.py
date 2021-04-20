'''
'''

import glob
import os
from xml.dom.minidom import parse

dir_xml = 'data/toxian20210419/*'

Epithelial = 0
Lymphocyte = 0
Neutrophile = 0
Others = 0
Zgnm = 0
Iud = 0
Agc = 0
Sus = 0
Hsil = 0
A = 0
Middle = 0
Gland = 0
Koilocyte = 0
Clusters = 0
Basical = 0
Small = 0
Hs = 0
Koilocyte_cell1 = 0
Blood = 0
Ncc = 0
Atr = 0
Clue = 0
Can = 0
Acti = 0
Tri = 0
Fun = 0
Bal = 0

Lvpao = 0
xf = 0
Ca = 0
Xm = 0

sub_dir_xml = glob.glob(dir_xml)

for each_sub_dir_xml in sub_dir_xml:
    path_xml = os.path.join(each_sub_dir_xml, '*.xml')
    list_path_xml = glob.glob(path_xml)

    if not list_path_xml:
        continue

    for each_path_xml in list_path_xml:

        # 解析.xml
        tree = parse(each_path_xml)
        root_node = tree.documentElement
        list_object = root_node.getElementsByTagName('object')

        for each_object in list_object:
            name_object = each_object.getElementsByTagName('name')[0].childNodes[0].data

            if name_object == 'Epithelial cell':
                Epithelial += 1

            elif name_object == 'Lymphocyte':
                Lymphocyte += 1

            elif name_object == 'Neutrophile':
                Neutrophile += 1

            elif name_object == 'Others':
                Others += 1

            elif name_object == 'Zgnm':
                Zgnm += 1

            elif name_object == 'Iud':
                Iud += 1

            elif name_object == 'Agc':
                Agc += 1

            elif name_object == 'Sus':
                Sus += 1

            elif name_object == 'Hsil':
                Hsil += 1

            elif name_object == 'A-H':
                A += 1

            elif name_object == 'Middle cell':
                Middle += 1

            elif name_object == 'Gland cell':
                Gland += 1

            elif name_object == 'Koilocyte cell':
                Koilocyte += 1

            elif name_object == 'Clusters':
                Clusters += 1

            elif name_object == 'Basical cell':
                Basical += 1

            elif name_object == 'Small cell':
                Small += 1

            elif name_object == 'Hs':
                Hs += 1

            elif name_object == 'Koilocyte cell1':
                Koilocyte_cell1 += 1

            elif name_object == 'Blood':
                Blood += 1

            elif name_object == 'Ncc':
                Ncc += 1

            elif name_object == 'Atr':
                Atr += 1

            elif name_object == 'Clue':
                Clue += 1

            elif name_object == 'Can':
                Can += 1

            elif name_object == 'Acti':
                Acti += 1

            elif name_object == 'Tri':
                Tri += 1

            elif name_object == 'Fun':
                Fun += 1

            elif name_object == 'Bal':
                Bal += 1

            # xml中不存在
            elif name_object == 'Lvpao':
                Lvpao += 1

            elif name_object == 'XF':
                xf += 1

            elif name_object == 'Ca':
                Ca += 1

            elif name_object == 'Xm':
                Xm += 1

            else:
                print('no such name: {name}'.format(name=name_object))

print('----------------------------------------')
print(
    'Epithelial cell'.ljust(20), '{Epithelial:>8d}'.format(Epithelial=Epithelial))
print(
    'Lymphocyte'.ljust(20), '{Lymphocyte:>8d}'.format(Lymphocyte=Lymphocyte))
print(
    'Neutrophile'.ljust(20), '{Neutrophile:>8d}'.format(Neutrophile=Neutrophile))
print(
    'Others'.ljust(20), '{Others:>8d}'.format(Others=Others))
print(
    'Zgnm'.ljust(20), '{Zgnm:>8d}'.format(Zgnm=Zgnm))
print('----------------------------------------')
print(
    'Iud'.ljust(20), '{Iud:>8d}'.format(Iud=Iud))
print(
    'Agc'.ljust(20), '{Agc:>8d}'.format(Agc=Agc))
print(
    'Sus'.ljust(20), '{Sus:>8d}'.format(Sus=Sus))
print(
    'Hsil'.ljust(20), '{Hsil:>8d}'.format(Hsil=Hsil + Ca))
print(
    'A-H'.ljust(20), '{A:>8d}'.format(A=A))
print('----------------------------------------')
print(
    'Middle cell'.ljust(20), '{Middle:>8d}'.format(Middle=Middle))
print(
    'Gland cell'.ljust(20), '{Gland:>8d}'.format(Gland=Gland))
print(
    'Koilocyte cell'.ljust(20), '{Koilocyte:>8d}'.format(Koilocyte=Koilocyte))
print(
    'Clusters'.ljust(20), '{Clusters:>8d}'.format(Clusters=Clusters))
print(
    'Basical cell'.ljust(20), '{Basical:>8d}'.format(Basical=Basical))
print('----------------------------------------')
print(
    'Small cell'.ljust(20), '{Small:>8d}'.format(Small=Small))
print(
    'Hs'.ljust(20), '{Hs:>8d}'.format(Hs=Hs))
print(
    'Koilocyte cell1'.ljust(20), '{Koilocyte_cell1:>8d}'.format(Koilocyte_cell1=Koilocyte_cell1))
print(
    'Blood'.ljust(20), '{Blood:>8d}'.format(Blood=Blood))
print(
    'Ncc'.ljust(20), '{Ncc:>8d}'.format(Ncc=Ncc))
print('----------------------------------------')
print(
    'Atr'.ljust(20), '{Atr:>8d}'.format(Atr=Atr))
print(
    'Clue'.ljust(20), '{Clue:>8d}'.format(Clue=Clue))
print(
    'Can'.ljust(20), '{Can:>8d}'.format(Can=Can))
print(
    'Acti'.ljust(20), '{Acti:>8d}'.format(Acti=Acti))
print(
    'Tri'.ljust(20), '{Tri:>8d}'.format(Tri=Tri))
print('----------------------------------------')
print(
    'Fun'.ljust(20), '{Fun:>8d}'.format(Fun=Fun))
print(
    'Bal'.ljust(20), '{Bal:>8d}'.format(Bal=Bal))

# xml中不存在
print(
    'SUS'.ljust(20), '{Lvpao:>8d}'.format(Lvpao=Lvpao))
print(
    'XF'.ljust(20), '{xf:>8d}'.format(xf=xf))
# print(
#     'Hsil'.ljust(20), '{Ca}'.format(Ca=Ca))
print(
    'Xm'.ljust(20), '{Xm:>8d}'.format(Xm=Xm))
print('----------------------------------------')
