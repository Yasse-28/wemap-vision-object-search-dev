/**
 * The one control that assigns a detection — or a whole cluster — to a group.
 *
 * A `<select>` rather than a text field: the annotation is only worth anything if the
 * same object gets the same name twice, and typing it out invites "extincteur" and
 * "extincteur " to become two objects. Creating a name stays possible, it just costs
 * one extra step.
 */

import type { GroupName } from "./groupAnnotations";

const NEW_GROUP = "\u0000new";
const NOT_AN_OBJECT = "\u0000reject";
const UNSET = "\u0000unset";

export function GroupPicker(props: {
  /** Current label; `undefined` means the detection has never been annotated. */
  value: GroupName | undefined;
  names: string[];
  disabled?: boolean;
  ariaLabel: string;
  /** Compact form for a detection card; the cluster header uses the wide one. */
  compact?: boolean;
  onAssign: (groupName: GroupName) => void;
  onClear: () => void;
}) {
  const current =
    props.value === undefined ? UNSET : props.value === null ? NOT_AN_OBJECT : props.value;

  return (
    <select
      className={`group-picker${props.compact ? " group-picker--compact" : ""}${
        props.value === undefined ? "" : " is-annotated"
      }`}
      aria-label={props.ariaLabel}
      title={props.ariaLabel}
      disabled={props.disabled}
      value={current}
      onChange={(event) => {
        const choice = event.target.value;
        if (choice === UNSET) {
          props.onClear();
          return;
        }
        if (choice === NOT_AN_OBJECT) {
          props.onAssign(null);
          return;
        }
        if (choice === NEW_GROUP) {
          const name = window.prompt("Name of the object this detection belongs to:");
          if (name?.trim()) {
            props.onAssign(name.trim());
          }
          return;
        }
        props.onAssign(choice);
      }}
    >
      <option value={UNSET}>— unannotated</option>
      {props.names.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
      <option value={NOT_AN_OBJECT}>not an object</option>
      <option value={NEW_GROUP}>new group…</option>
    </select>
  );
}

export default GroupPicker;
