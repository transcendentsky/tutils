import SimpleITK as sitk


# ------------------ Spacing  ------------------
def resampleImage(Image:sitk.SimpleITK.Image, SpacingScale=None, NewSpacing=None, NewSize=None, Interpolator=sitk.sitkLinear)->sitk.SimpleITK.Image:
    """
    Author: Pengbo Liu
    Function: resample image to the same spacing
    Params:
        Image, SITK Image
        SpacingScale / NewSpacing / NewSize , are mutual exclusive, independent.
    """
    Size = Image.GetSize()
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()

    if not SpacingScale is None and NewSpacing is None and NewSize is None:
        NewSize = [int(Size[0]/SpacingScale),
                   int(Size[1]/SpacingScale),
                   int(Size[2]/SpacingScale)]
        NewSpacing = [Spacing[0]*SpacingScale,
                      Spacing[1]*SpacingScale,
                      Spacing[2]*SpacingScale]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1],  NewSpacing[2]))
    elif not NewSpacing is None and SpacingScale is None and NewSize is None:
        NewSize = [int(Size[0] * Spacing[0] / NewSpacing[0]),
                   int(Size[1] * Spacing[1] / NewSpacing[1]),
                   int(Size[2] * Spacing[2] / NewSpacing[2])]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1], NewSpacing[2]))
    elif not NewSize is None and SpacingScale is None and NewSpacing is None:
        NewSpacing = [Spacing[0]*Size[0] / NewSize[0],
                      Spacing[1]*Size[1] / NewSize[1],
                      Spacing[2]*Size[2] / NewSize[2]]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0],Spacing[1],Spacing[2],NewSpacing[0],NewSpacing[1],NewSpacing[2]))


    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    Resample.SetOutputSpacing(NewSpacing)
    Resample.SetInterpolator(Interpolator)
    NewImage = Resample.Execute(Image)

    return NewImage    

def usage():
    image = resampleImage(image, NewSpacing=new_spacing)