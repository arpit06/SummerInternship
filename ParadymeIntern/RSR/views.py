# -*- coding: utf-8 -*-
from .models import *
import docx2txt
from django.utils import timezone

# Create your views here.
#=======
# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.contrib.auth.decorators import login_required
from django.forms import ModelForm

from RSR.models import *
from RSR.forms import *
from django.shortcuts import get_object_or_404
from django.contrib.auth import logout
from .filters import *
###Search #
from django.db.models import Q
from RSR.persondetails import Detail
from RSR.persondetails2 import Detail2
from django.views.generic.edit import UpdateView


### json Parsing ##
import json

###TESTING OCR
#from PIL import Image
#from wand.image import Image as IMG
#import pytesseract
#import textract
###
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx_viewer import Viewer
import pandas as pd
import nxviz as nv
import numpy as np
import matplotlib.pyplot as plt



def logout_page(request):
    logout(request)
    return HttpResponseRedirect('/')

@login_required
def main(request):
    return render(request, 'main.html')

#def get_string(name):
#    img=Image.open(name)
#    utf8_text = pytesseract.image_to_string(img)
#    utf8_text = str(utf8_text.encode('ascii', 'ignore'))
#    return utf8_text


@login_required
def uploaddoc(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            temp_doc = Document(docfile=request.FILES['docfile'])

            #temp_doc.firstname = Document(docfile=request.POST.get('firstname'))
            #temp_doc.lastname = Document(docfile=request.POST.get('lastname'))
            #temp_doc.type = Document(docfile=request.POST.get('type'))
            temp_doc.firstname = request.POST['firstname']
            temp_doc.lastname = request.POST['lastname']
            temp_doc.type = request.POST['type']
            temp_doc.uploaduser = request.user.username

            temp_doc.save()

            if ".doc" in temp_doc.docfile.path:
                print (temp_doc.docfile.path)
                temp_doc.docfile.wordstr = parse_word_file(temp_doc.docfile.path)
                print (temp_doc.docfile.wordstr)
                temp_doc.save(update_fields=['wordstr'])

            else:

            #    temp_doc.docfile.wordstr = textract.process(temp_doc.docfile.path)

            #    if len(temp_doc.docfile.wordstr) < 50:
                img=IMG(filename=temp_doc.docfile.path,resolution=200)

                img.save(filename='temp.jpg')
                utf8_text = get_string('temp.jpg')
                os.remove('temp.jpg')

                print (utf8_text)
                temp_doc.docfile.wordstr = utf8_text
                temp_doc.save(update_fields=['wordstr'])

                print (temp_doc.docfile.wordstr)
                temp_doc.save(update_fields=['wordstr'])

            parsed_json  = parse_file(temp_do.docfile.wordstr)
            #json testing#
            #check for json file, wont be needed as parsing will return json#
            if ".json" in temp_doc.docfile.path:
                #either load json, or recieve json file
                js = json.load(open(temp_doc.docfile.path))
                #iterate through json file

                #initialize person out side of for loop/if statements so we can use it later
                person = Person(Name="temp")
                for label in js:

                    #Checking Labels to see which table to create
                    if label == "person":
                        for key in js[label]:
                            if key == "name":
                                person.Name = js[label][key]
                            elif key == "email":
                                person.Email = js[label][key]
                            elif key == "address":
                                person.Address = js[label][key]
                            elif key == "zipcode":
                                person.ZipCode = js[label][key]
                            elif key == "state":
                                person.State = js[label][key]
                            elif key == "phone":
                                person.PhoneNumber = js[label][key]
                            elif key == "linkedin":
                                person.Linkedin = js[label][key]
                            elif key == "github":
                                person.GitHub = js[label][key]
                        person.Resume = temp_doc.docfile
                        person.TypeResume = temp_doc.type
                        person.save()


                    elif label == "skills":
                        for key in js[label]:
                            #check to see if skill exists
                            query_set=Skills.objects.all()
                            query_set=query_set.filter(Name__icontains=key["skill"])
                            #if skill does not exist create skill
                            if not query_set:
                                query_set = Skills(Name = key["skill"])
                                query_set.save()
                            #if skill does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            skill_to_person = PersonToSkills(SkillsID = query_set, PersonID = person,YearsOfExperience = key["YearsOfExperience"])
                            skill_to_person.save()

                    elif label == "work":
                        for key in js[label]:
                            #check to see if company exists
                            query_set=Company.objects.all()
                            query_set=query_set.filter(Name__icontains=key["company"])
                            #if company does not exist create skill
                            if not query_set:
                                query_set = Company(Name = key["company"])
                                query_set.save()
                            #if company does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            #intermediary table stuff
                            company_to_person = PersonToCompany(CompanyID = query_set, PersonID = person,
                                Title = key["title"],
                                ExperienceOnJob = key["experience"],
                                StartDate = key["startDate"],
                                EndDate = key["endDate"],
                                Desc = key["summary"])
                            company_to_person.save()

                    elif label == "education":
                        for key in js[label]:
                            #check to see if School exists
                            query_set=School.objects.all()
                            query_set=query_set.filter(Name__icontains=key["school"]["name"]).filter(DegreeLevel = key["school"]["degreeLevel"])
                            #if School does not exist create skill
                            if not query_set:
                                query_set = School(Name = key["school"]["name"], DegreeLevel = key["school"]["degreeLevel"])
                                query_set.save()
                            #if School does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]

                            # NOW DO MAJOR
                            query_set_1=Major.objects.all()
                            query_set_1=query_set_1.filter(Name__icontains=key["major"]["major"]).filter(Dept__icontains = key["major"]["dept"]).filter(MajorMinor__icontains = key["major"]["major/minor"])
                            if not query_set_1:
                                query_set_1 = Major(Name = key["major"]["major"], Dept = key["major"]["dept"], MajorMinor = key["major"]["major/minor"])
                                query_set_1.save()
                            #if School does exist, grab first match from queryset
                            else:
                                query_set_1 = query_set_1[0]

                            #intermediary table stuff
                            ed_to_person = PersonToSchool(SchoolID = query_set, PersonID = person, MajorID = query_set_1,
                                GPA = key["GPA"],
                                GradDate = key["gradDate"])
                            ed_to_person.save()


                    elif label == "sideprojects":
                        for key in js[label]:
                            #check to see if project exists
                            query_set=SideProject.objects.all()
                            query_set=query_set.filter(Name__icontains=key["name"])
                            #if project does not exist create project
                            if not query_set:
                                query_set = SideProject(Name = key["name"])
                                query_set.save()
                            #if project does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            #intermediary table stuff
                            project_to_person = PersonToSide(SideID = query_set, PersonID = person, Desc = key["description"])
                            project_to_person.save()

                    elif label == "award":
                        for key in js[label]:
                            #check to see if Award exists
                            query_set=Awards.objects.all()
                            query_set=query_set.filter(Name__icontains=key["name"])
                            #if Award does not exist create Award
                            if not query_set:
                                query_set = Awards(Name = key["name"])
                                query_set.save()
                            #if Award does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            #intermediary table stuff
                            awards_to_person = PersonToAwards(AwardID = query_set, PersonID = person, Desc = key["description"])
                            awards_to_person.save()

                    elif label == "clearance":
                        query_set = Clearance.objects.all()
                        query_set = query_set.filter(ClearanceLevel = js[label]["level"])
                        if not query_set:
                            query_set = Clearance(ClearanceLevel=js[label]["level"])
                            query_set.save()
                        else:
                            query_set = query_set[0]
                        cl_to_person = PersonToClearance(PersonID=person, ClearanceLevel = query_set)
                        cl_to_person.save()

                    elif label == "languages":
                        for key in js[label]:
                            # check to see if language exists
                            query_set = LanguageSpoken.objects.all()
                            query_set = query_set.filter(Language__icontains=key["language"])
                            # if language does not exist create language
                            if not query_set:
                                query_set = LanguageSpoken(Language=key["language"])
                                query_set.save()
                            # if language does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            # intermediary table stuff
                            language_to_person = PersonToLanguage(LangID=query_set, PersonID=person)
                            language_to_person.save()

                    elif label == "clubs":
                        for key in js[label]:
                            # check to see if club exists
                            query_set = Clubs_Hobbies.objects.all()
                            query_set = query_set.filter(Name__icontains=key["name"])
                            # if club does not exist create club
                            if not query_set:
                                query_set = Clubs_Hobbies(Name=key["name"])
                                query_set.save()
                            # if club does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            # intermediary table stuff
                            club_to_person = PersonToClubs_Hobbies(CHID=query_set, PersonID=person, Desc=key["description"])
                            club_to_person.save()

                    elif label == "volunteering":
                        for key in js[label]:
                            # check to see if volunteer exists
                            query_set = Volunteering.objects.all()
                            query_set = query_set.filter(Name__icontains=key["name"])
                            # if volunteer does not exist create volunteer
                            if not query_set:
                                query_set = Volunteering(Name=key["name"])
                                query_set.save()
                            # if volunteer does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            # intermediary table stuff
                            volunteer_to_person = PersonToVolunteering(VolunID=query_set, PersonID=person, Desc=key["description"])
                            volunteer_to_person.save()

                    elif label == "course":
                        for key in js[label]:
                            # check to see if course exists
                            query_set = Coursework.objects.all()
                            query_set = query_set.filter(Name__icontains=key["name"])
                            # if course does not exist create course
                            if not query_set:
                                query_set = Coursework(Name=key["name"])
                                query_set.save()
                            # if course does exist, grab first match from queryset
                            else:
                                query_set = query_set[0]
                            # intermediary table stuff
                            course_to_person = PersonToCourse(CourseID=query_set, PersonID=person,Desc=key["description"])
                            course_to_person.save()


            return HttpResponseRedirect(reverse('RSR:uploaddoc'))
    else:
        form = DocumentForm()

    documents = Document.objects.all()
    return render(request,'index.html',{'documents': documents, 'form': form})

def person_edit(request, person_id):
	instance = get_object_or_404(Person, id=person_id)
	form = PersonForm(request.POST or None, instance=instance)


	if form.is_valid():
		form.save()

		return HttpResponseRedirect(reverse('RSR:detail', args=[instance.pk]))
	context = {
		'form' : form,
		'pk' : person_id,
		'person':instance
	}


	return render(request, 'person_update_form.html', context)




@login_required
def ocr (request):
    return render(request, 'ocr.html')

@login_required
def parsing(request):
    return render(request, 'parsing.html')


@login_required
def search(request):
    query_set = Person.objects.all()
    query = request.GET.get("q")
    if query:
        query_set=query_set.filter(Name__icontains=query)
    # The filtered query_set is then put through more filters from django
    personFilter = PersonFilter(request.GET, query_set)
    #make_LA_Connections(request.GET,personFilter.queryset)
    return render(request, 'SearchExport/search.html', {'personFilter': personFilter})

def personSearch(cat,key,person):
    if cat == 'YearOfExperienceForSkill':
        if len(PersonToSkills.objects.filter(PersonID = person.pk).filter(YearsOfExperience = key)) != 0:
            return key
        else:
            return None
    elif cat == 'GraduateDate':
        if PersonToSchool.objects.filter(PersonID = person.pk).filter(GradDate = key) != 0:
            return key
    elif cat == 'CompanyWorked':
        try:
            return PersonToCompany.objects.filter(PersonID = person.pk).get(CompanyID = key).CompanyID
        except PersonToCompany.DoesNotExist:
            return None
    elif cat == 'Award':
        try:
            return PersonToAwards.objects.filter(PersonID = person.pk).get(AwardID = key).AwardID
        except PersonToAwards.DoesNotExist:
            return None
    elif cat == 'ProfessionalDevelopment':
        print(PersonToProfessionalDevelopment.objects.filter(PersonID = p.pk))
    elif cat == 'GPAub':
        try:
            return PersonToSchool.objects.filter(GPA__lte = key).get(PersonID = person.pk).GPA
        except PersonToSchool.DoesNotExist:
            return None
    elif cat == 'GPAlb':
        try:
            return PersonToSchool.objects.filter(GPA__gte = key).get(PersonID = person.pk).GPA
        except PersonToSchool.DoesNotExist:
            return None
    elif cat == 'SchoolAttend':
        try:
            return PersonToSchool.objects.filter(PersonID = person.pk).get(SchoolID = key).SchoolID
        except PersonToSchool.DoesNotExist:
            return None
    elif cat == 'Club_Hobby':
        try:
            return PersonToClubs_Hobbies.objects.filter(CHID = key).get(PersonID = person.pk).CHID
        except PersonToClubs_Hobbies.DoesNotExist:
            return None
    elif cat == 'Skills':
        try:
            return PersonToSkills.objects.filter(PersonID = person.pk).get(SkillsID = key).SkillsID
        except PersonToSkills.DoesNotExist:
            return None
    elif cat == 'DegreeLevel':
        schools = PersonToSchool.objects.filter(PersonID = person.pk)
        for s in schools:
            if s.SchoolID.DegreeLevel == key:
                return key
        return None
    elif cat == 'Major':
        try:
            return PersonToSchool.objects.filter(MajorID = key).get(PersonID = person.pk).MajorID
        except PersonToSchool.DoesNotExist:
            return None
    elif cat == 'SecurityClearance':
        if PersonToClearance.objects.filter(PersonID = person.pk).filter(ClearanceLevel = key) != 0:
            return key
        else:
            return None
    elif cat == 'Language':
        try:
            return PersonToLanguage.objects.filter(LangID = key).get(PersonID = person.pk).LangID
        except PersonToLanguage.DoesNotExist:
            return None
    elif cat == 'Volunteering':
        if PersonToVolunteering.objects.filter(PersonID = person.pk).filter(VolunID = Volunteering.objects.get(Name = key).pk)!= 0:
            return key
        else:
            return None
    elif cat == 'Title':
        if PersonToCompany.objects.filter(PersonID = person.pk).filter(Title = key)!= 0:
            return key
        else:
            return None

@login_required
def detail(request,pk):
       # Get the current person object using pk or id
    person = get_object_or_404(Person, pk=pk)
    related_obj_list=Detail(person)

    detail_dic = Detail2(person)
    School = detail_dic['PersonToSchool']
    Course = detail_dic['PersonToCourse']
    Pro = detail_dic['PersonToProfessionalDevelopment']
    Side = detail_dic['PersonToSide']
    Skills = detail_dic['PersonToSkills']
    Language = detail_dic['PersonToLanguage']
    Clearance = detail_dic['PersonToClearance']
    Company = detail_dic['PersonToCompany']
    Clubs = detail_dic['PersonToClubs_Hobbies']
    Volunteer = detail_dic['PersonToVolunteering']
    Award = detail_dic['PersonToAwards']
    context = {
                'person':person,
                'list': related_obj_list,
                'school':School,
                'course':Course,
                'pro':Pro,
                'side':Side,
                'skills':Skills,
                'language':Language,
                'clearance':Clearance,
                'company':Company,
                'clubs':Clubs,
                'volunteer':Volunteer,
                'award':Award,
                }

    return render(request, 'SearchExport/detail.html', context)





@login_required
def user_acc_cont (request):
    return render(request, 'acc_cont.html')

@login_required
def export(request):
    return render (request, 'export.html')

@login_required
def linkanalysis(request):
    query_set = Person.objects.all()
    query = request.GET.get("q")
    if query:
        query_set=query_set.filter(Name__icontains=query)
    # The filtered query_set is then put through more filters from django
    personFilter = PersonFilter(request.GET, query_set)
    arr = []
    print(request.GET)
    for key in request.GET:
        if request.GET[key] !='':
            arr.append(key)
    arr.append('Person')
    df = pd.DataFrame(columns = arr)
    arr.remove('Person')
    for person in personFilter.queryset:
        new_entry = {}
        new_entry['Person'] = person.Name
        for key in arr:
            new_entry[key] = personSearch(key,request.GET[key],person)
            #new_entry[key] = request.GET[key]
        df = df.append(new_entry,ignore_index = True)
    print(personFilter.queryset)
    make_LA(df)
    return render(request, 'SearchExport/search.html', {'personFilter': personFilter})


def make_LA(df):
    Columns=list(df)
    print(len(Columns))
    i=0
    for column in Columns:
        G=nx.from_pandas_dataframe(df,df.columns[0],column)
        G.add_edges_from(G.edges())
        for i in range(1,len(Columns)-1):
            H=nx.from_pandas_dataframe(df,df.columns[i],column)
            G.add_edges_from(H.edges())
    print(df)
    nx.info(G)

    Nodes = G.number_of_nodes()
    Edges = G.number_of_edges()

    pos = nx.spring_layout(G,k=0.9,iterations=1, scale=2)
    #pos = nx.shell_layout(G)
    #pos = nx.nx_pydot.graphviz_layout(G, prog = 'dot')
    N = Nodes
    E = Edges
    #pos = nx.random_layout(G,scale=2)
    plt.figure(1,figsize=(20, 20), dpi=100)
    colors_nodes = np.random.rand(N)
    colors_edges = np.random.rand(E)
    nx.draw_networkx(G,pos, node_color=colors_nodes, edge_color=colors_edges, arrows=True, node_size = 4500,
                 node_shape='p', alpha= 1,linewidths= 10, clip_on=1,)
    nx.draw_networkx_labels(G,pos, font_size=12, font_color= 'R')
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    plt.show()



@login_required
def uploadlist (request):
   # documents = Document.objects.filter(firstname = Document.firstname).filter(lastname = Document.lastname).filter(type = Document.type).filter(docfile = Document.docfile)
    documents = UploadListFilter(request.GET,queryset = Document.objects.all())
    #documents = Document.objects.all()
    context ={'documents':documents}
    return render(request,'uploadlist.html',context)

def listdelete(request, template_name='uploadlist.html'):
    docId = request.POST.get('docfile', None)
    documents = get_object_or_404(Document, pk=docId)
    if request.method == 'POST':


        documents.delete()
        return HttpResponseRedirect(reverse('RSR:uploadlist'))

    return render(request, template_name, {'object': documents})


def parse_word_file(filepath):
	parsed_string = docx2txt.process(filepath)
	return parsed_string
def parse_file(res_string):
    return 1
