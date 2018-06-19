# Tasks

A task is an assignment that is given to the system that is being controlled by
Assist. A task is represented like a function call. The type of task (e.g.
switching on/off a light) is represented by which function is being called and
the specifics are represented by its arguments. For example turning on the
bedroom light is represented by `switch(bedroom_light, on)`.

All possible tasks that can be performed by a system are represented by the
semantic structure. The semantic structure contains a list of task types
and a list of argument types.

## Task types

A task type is defined by a name and a list of typed arguments. For example,
the task type of switching a light has the name switch and has two
arguments: the light to turn on, which is of type light and the state of the
light which is of type light_state.

## Argument types

An argument type is defined by a name, a supertype and possibly other
properties. The supertype determines how the semantic coder will interpret the
argument and how it will be encoded. An example of a supertype is an Enumerable
which defines a set of possible values the argument can take. For example
the argument type light has the name light, is an Enumerable and could for
example take the values bedroom_light and kitchen_light.

## Semantic structure file

The semantic structure file contains the semantic structure. It is in the xml
format. As an example a device that can turn on/off two lights and open/close
three doors looks something like this:

```
<structure>
  <types>
    <light_state supertype="enumerable">
      on
      off
    </light_state>
    <light supertype="enumerable">
      bedroom_light
      kitchen_light
    </light>
    <door supertype="enumerable">
      bedroom_door
      kitchen_door
      front_door
    </door>
  </types>
  <tasks>
    <switch target_light="light" state="light_state"/>
    <open target_door="door"/>
    <close target_door="door"/>
  </tasks>
</structure>
```

## Task representation

Using the semantic structure the task can be represented as a string containing
the task type name and the arguments. The task to turn on the bedroom light
would be represented as
`<switch target_light="bedroom_light" state="on"/>`.

# Coder

A coder can encode a task string into a vector representation using the semantic
structure. It takes a task string as input and returns the vector
representation. A coder can also decode an (esimation of) a  semantic vector
representation into the (most likely) task. To create your own coder
you can inherit from the general Coder class defined in
coder.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in
coder_factory.py.
